from model import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Build the model
model = build_model((256, 256, 3))

# Setup data generators
datagen = ImageDataGenerator(validation_split=0.2)

train_generator = datagen.flow_from_directory(
    r'C:\Users\soura\Downloads\archive (1)\TrainingSet',
    target_size=(256, 256),
    batch_size=32,
    subset='training',
    class_mode='input'
)

validation_generator = datagen.flow_from_directory(
    r'C:\Users\soura\Downloads\archive (1)\TrainingSet',
    target_size=(256, 256),
    batch_size=32,
    subset='validation',
    class_mode='input'
)

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Save the model
model.save('enhancement_model.h5')
