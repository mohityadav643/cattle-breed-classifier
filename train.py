import os
print("🚀 STARTING TRAINING...")

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# ======================
# PATHS
# ======================
train_dir = "dataset/train"
val_dir = "dataset/val"

print("Train folders:", os.listdir(train_dir))
print("Val folders:", os.listdir(val_dir))

# ======================
# SETTINGS
# ======================
IMG_SIZE = 224
BATCH_SIZE = 32

# ======================
# DATA GENERATORS
# ======================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ======================
# MODEL (MobileNetV2)
# ======================
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False  # freeze base

# Custom head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(train_data.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=outputs)

# ======================
# COMPILE
# ======================
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ======================
# TRAIN
# ======================
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    verbose=1
)

# ======================
# SAVE MODEL
# ======================
model.save("model.h5")

print("✅ TRAINING DONE")