import tensorflow as tf
from tensorflow.keras import layers, models

def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')  # Output layer
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def load_model(model_path):
    return tf.keras.models.load_model(model_path)
