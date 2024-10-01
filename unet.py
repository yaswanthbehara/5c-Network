from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def nested_unet(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    return Model(inputs, outputs)
