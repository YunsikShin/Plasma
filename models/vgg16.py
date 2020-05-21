import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

class vgg16:
    def __init__(self, input_shape):
        print('[i]  Model Module. Vgg16')
        self.input_shape = input_shape 
        self.get_model()

    def get_model(self):
        filters = [64, 128, 256, 512, 512]
        input_tensor = tf.keras.Input(shape = self.input_shape)
        x = Conv2D(filters = filters[0], kernel_size = (4, 3), activation = 'relu',
                   strides = (1, 1), padding = 'valid', name = 'conv1_1')(input_tensor)
        x = Conv2D(filters = filters[0], kernel_size = (1, 3), activation = 'relu',
                   strides = (1, 1), padding = 'valid', name = 'conv1_2')(x)
        x = MaxPool2D(pool_size = (1, 2), padding = 'valid')(x)
        x = Conv2D(filters = filters[1], kernel_size = (1, 3), activation = 'relu',
                   strides = (1, 1), padding = 'valid', name = 'conv2_1')(x)
        x = Conv2D(filters = filters[1], kernel_size = (1, 3), activation = 'relu',
                   strides = (1, 1), padding = 'valid', name = 'conv2_2')(x)
        x = MaxPool2D(pool_size = (1, 2), padding = 'valid')(x)
        x = Conv2D(filters = filters[2], kernel_size = (1, 3), activation = 'relu',
                   strides = (1, 1), padding = 'valid', name = 'conv3_1')(x)
        x = Conv2D(filters = filters[2], kernel_size = (1, 3), activation = 'relu',
                   strides = (1, 1), padding = 'valid', name = 'conv3_2')(x)
        x = MaxPool2D(pool_size = (1, 2), padding = 'valid')(x)
        x = Conv2D(filters = filters[3], kernel_size = (1, 3), activation = 'relu',
                   strides = (1, 1), padding = 'valid', name = 'conv4_1')(x)
        x = Conv2D(filters = filters[3], kernel_size = (1, 3), activation = 'relu',
                   strides = (1, 1), padding = 'valid', name = 'conv4_2')(x)
        x = MaxPool2D(pool_size = (1, 2), padding = 'valid')(x)
        x = Conv2D(filters = filters[4], kernel_size = (1, 3), activation = 'relu',
                   strides = (1, 1), padding = 'valid', name = 'conv5_1')(x)
        x = Conv2D(filters = filters[4], kernel_size = (1, 3), activation = 'relu',
                   strides = (1, 1), padding = 'valid', name = 'conv5_2')(x)
        x = MaxPool2D(pool_size = (1, 2), padding = 'valid')(x)
        x = Flatten()(x)
        x = Dense(units = 1024, activation = 'relu', name = 'fc1')(x)
        x = Dense(units = 1024, activation = 'relu', name = 'fc2')(x)
        x = Dense(units = 2, activation = None, name = 'logit_layer')(x)
        self.model = tf.keras.Model(inputs = input_tensor, outputs = x)

