from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input

resnet = ResNet50(weights="imagenet", include_top=False,
                         input_tensor=Input(shape=(224, 224, 3)))

resnet.save("resnet50_weights.h5")