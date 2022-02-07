from tensorflow.keras.applications import ResNet50, InceptionV3
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense
from tensorflow.keras.models import Model

def get_model(num_classes):
    res