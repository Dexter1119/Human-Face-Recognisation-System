# ============================================================
# FILE NAME : face_recog.py
# Description : CelebA Face Classification (Optimized for Laptop)
# Author : Pradhumnya Changdev Kalsait
# Date : 26/04/26
# ============================================================

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pandas as pd
import numpy as np
import cv2
import os

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

IMG_HEIGHT = 64
IMG_WIDTH = 64

DATA_PATH = "img_align_celeba"
CSV_PATH = "list_attr_celeba.csv"
MODEL_PATH = "face_model.keras"


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    n_samples = int(input("Enter number of samples (e.g., 2000–10000): "))

    # Safe adaptive config
    if n_samples <= 3000:
        batch_size = 8
        epochs = 12
    elif n_samples <= 7000:
        batch_size = 16
        epochs = 12
    else:
        batch_size = 32
        epochs = 15

    print(f"\nUsing batch_size={batch_size}, epochs={epochs}")

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    df['Smiling'] = df['Smiling'].map({1: "Smiling", -1: "Not_Smiling"})

    # Filter existing images
    existing = set(os.listdir(DATA_PATH))
    df = df[df['image_id'].isin(existing)]
    df = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df['Smiling']
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['Smiling']
    )

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )

    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, DATA_PATH, x_col='image_id', y_col='Smiling',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size, class_mode='binary'
    )

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes
    )
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)

    val_gen = eval_datagen.flow_from_dataframe(
        val_df, DATA_PATH, x_col='image_id', y_col='Smiling',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size, class_mode='binary', shuffle=False
    )

    test_gen = eval_datagen.flow_from_dataframe(
        test_df, DATA_PATH, x_col='image_id', y_col='Smiling',
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=batch_size, class_mode='binary', shuffle=False
    )

    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )

    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    early = EarlyStopping(patience=3, restore_best_weights=True, monitor='val_accuracy', mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[early, reduce_lr],
        class_weight=class_weights
    )

    loss, acc = model.evaluate(test_gen)
    print(f"\nTest Accuracy: {acc:.4f}")

    # ============================================================
    # EVALUATION (FIXED)
    # ============================================================

    test_gen.reset()

    y_prob = model.predict(test_gen, verbose=0)

    threshold = 0.6
    y_pred = (y_prob > threshold).astype(int).flatten()

    y_true = test_gen.classes

    print("\nSanity Check:")
    print("Samples:", len(y_true))
    print("Pred shape:", y_pred.shape)
    print("Class indices:", test_gen.class_indices)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Not Smiling","Smiling"],
                yticklabels=["Not Smiling","Smiling"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=["Not Smiling","Smiling"]))

    model.save(MODEL_PATH)
    print("Model saved!")

# ============================================================
# IMAGE PREDICTION
# ============================================================

def predict_image():

    if not os.path.exists(MODEL_PATH):
        print("Model not found! Train first.")
        return

    model = load_model(MODEL_PATH)

    path = input("Enter image path: ")

    img = cv2.imread(path)
    if img is None:
        print("Invalid path!")
        return

    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    print("Smiling 😊" if pred >= 0.6 else "Not Smiling 😐")


# ============================================================
# WEBCAM (FIXED VERSION)
# ============================================================

def webcam():

    if not os.path.exists(MODEL_PATH):
        print("Model not found! Train first.")
        return

    model = load_model(MODEL_PATH)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    cap = cv2.VideoCapture(0)

    frame_count = 0
    pred_val = 0.5

    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x,y,w,h) in faces:
            face = frame[y:y+h, x:x+w]

            if face.size == 0:
                continue

            face = cv2.resize(face, (IMG_HEIGHT, IMG_WIDTH))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = preprocess_input(face.astype(np.float32))
            face = np.expand_dims(face, axis=0)

            if frame_count % 5 == 0:
                pred_val = model.predict(face, verbose=0)[0][0]

            label = "Smiling 😊" if pred_val > 0.6 else "Not Smiling 😐"

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(frame,label,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        frame_count += 1

        cv2.imshow("Webcam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ============================================================
# MENU
# ============================================================

def main():

    while True:
        print("\n==== FACE RECOGNITION SYSTEM ====")
        print("1. Train Model")
        print("2. Predict from Image")
        print("3. Webcam Detection")
        print("4. Exit")

        choice = input("Enter choice: ")

        if choice == '1':
            train_model()
        elif choice == '2':
            predict_image()
        elif choice == '3':
            webcam()
        elif choice == '4':
            break
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()