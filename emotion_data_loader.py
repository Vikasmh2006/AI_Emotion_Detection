from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image size used in FER dataset
img_size = 48
batch_size = 32

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Training data
train_data = train_datagen.flow_from_directory(
    "train",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training"
)

# Validation data
val_data = train_datagen.flow_from_directory(
    "train",
    target_size=(img_size, img_size),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

print("Classes:", train_data.class_indices)