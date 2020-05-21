#import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import utils.flags as flags_module
import models.vgg16 as model_module
import importlib
import numpy as np
import pdb


class train_class:
    def __init__(self, sys_flags, data_flags, train_flags):
        self.sys_flags = sys_flags
        self.data_flags = data_flags
        self.train_flags = train_flags
        dir_factory = importlib.import_module(sys_flags.module_factory)
        self.factory = dir_factory.factory_class(sys_flags, data_flags)
        model_class = model_module.vgg16(input_shape = (4, 256, 1))
        self.model = model_class.model
        self.initialization()

    def initialization(self):
        self.train_data = np.load(os.path.join(self.factory.dir_train_data))
        self.test_data = np.load(os.path.join(self.factory.dir_test_data))
        self.num_train_img = self.train_data.shape[0]
        self.num_test_img = self.test_data.shape[0]
        self.train_indices = np.arange(self.num_train_img)
        self.test_indices = np.arange(self.num_test_img)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.train_flags.learning_rate)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy(name = 'acc')
        self.train_batch_count = 0
        self.test_batch_count = 0

    def train_next_batch(self):
        batch_size = self.train_flags.batch_size
        self.max_batch_count = int(self.num_train_img / batch_size)
        if self.train_batch_count = 0:
            np.random.shuffle(self.train_indices)
        batch_img = np.zeros(shape = (batch_size, self.data_flags.Ls, self.data_flags.Ls_shift),
                             dtype = np.float32)
        batch_idx = self.train_indices[self.train_batch_count * batch_size:\
                                       (self.train_batch_count + 1)*batch_size]
        batch_data = self.train_data[batch_idx]



    


if __name__ == '__main__':
    sys_flags = flags_module.get_sys_flags()
    data_flags = flags_module.get_data_flags()
    train_flags = flags_module.get_train_flags()
    train_class(sys_flags = sys_flags, data_flags = data_flags, train_flags = train_flags)
