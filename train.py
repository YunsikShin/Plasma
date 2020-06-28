#import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import utils.flags as flags_module
import models.vgg16 as model_module
import importlib
import numpy as np
import tensorflow as tf
import pdb


class train_class:
    def __init__(self):
        self.sys_flags = flags_module.get_sys_flags()
        self.data_flags = flags_module.get_data_flags()
        self.train_flags = flags_module.get_train_flags()
        dir_factory = importlib.import_module(sys_flags.module_factory)
        self.factory = dir_factory.factory_class(sys_flags, data_flags)
        model_class = model_module.vgg16(input_shape = (4, 256, 1))
        self.model = model_class.model
        self.initialization()
        self.train_model()

    def initialization(self):
        self.train_data = np.load(os.path.join(self.factory.dir_train_data))
        self.test_data = np.load(os.path.join(self.factory.dir_test_data))
        self.test_abnormal_data = self.test_data[np.where(self.test_data[:, 1] == '1.0')[0]]
        self.test_normal_data = self.test_data[np.where(self.test_data[:, 1] == '0.0')[0]]
        
        self.num_train_img = self.train_data.shape[0]
        self.num_test_img = self.test_data.shape[0]
        self.train_indices = np.arange(self.num_train_img)
        self.test_indices = np.arange(self.num_test_img)
        self.test_normal_indices = np.arange(self.test_normal_data.shape[0])
        self.test_abnormal_indices = np.arange(self.test_abnormal_data.shape[0])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = self.train_flags.learning_rate)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.acc = tf.keras.metrics.SparseCategoricalAccuracy(name = 'acc')
        self.train_batch_count = 0
        self.test_batch_count = 0
        self.test_abnormal_batch_count = 0
        self.test_normal_batch_count = 0
        self.iter = 0
        self.epoch = 1
        self.test_epoch = 1

    def train_next_batch(self):
        batch_size = self.train_flags.batch_size
        self.max_batch_count = int(self.num_train_img / batch_size)
        if self.train_batch_count == 0:
            np.random.shuffle(self.train_indices)
        batch_img = np.zeros(shape = (batch_size, 4, self.data_flags.Ls),
                             dtype = np.float32)
        batch_idx = self.train_indices[self.train_batch_count * batch_size:\
                                       (self.train_batch_count + 1)*batch_size]
        batch_data = self.train_data[batch_idx]
        for i in range(batch_size):
            img = np.load(batch_data[i, 0])
            img = img[1:, :]
            batch_img[i, :, :] = img
        batch_img = np.expand_dims(batch_img, axis=3)
        batch_label = self.train_data[batch_idx, 1].astype(np.float).astype(np.int)
        self.train_batch_count += 1
        if self.train_batch_count == self.max_batch_count - 1:
            self.train_batch_count = 0
            self.epoch += 1
        return batch_img, batch_label

    def test_next_batch(self):
        batch_size = self.train_flags.batch_size
        batch_size = int(self.num_test_img / batch_size)
        max_batch_count = int(self.num_test_img / batch_size)
        if self.test_batch_count == 0:
            np.random.shuffle(self.test_indices)
        batch_img = np.zeros(shape = (batch_size, 4, self.data_flags.Ls),
                             dtype = np.float32)
        batch_idx = self.test_indices[self.test_batch_count * batch_size:\
                                      (self.test_batch_count + 1) * batch_size]
        batch_data = self.test_data[batch_idx]
        for i in range(batch_size):
            img = np.load(batch_data[i, 0])
            img = img[1:, :]
            batch_img[i, :, :] = img
        batch_img = np.expand_dims(batch_img, axis=3)
        batch_label = self.test_data[batch_idx, 1].astype(np.float).astype(np.int)
        self.test_batch_count += 1
        if self.test_batch_count == max_batch_count:
            self.test_batch_count = 0
            self.test_epoch += 1
        return batch_img, batch_label

    def test_next_normal_batch(self):
        batch_size = self.train_flags.batch_size
        max_batch_count = int(self.test_normal_data.shape[0] / batch_size)
        if self.test_abnormal_batch_count == 0:
            np.random.shuffle(self.test_normal_indices)
        batch_img = np.zeros(shape = (batch_size, 4, self.data_flags.Ls),
                             dtype = np.float32)
        batch_idx = self.test_normal_indices[self.test_abnormal_batch_count * batch_size:\
                                      (self.test_abnormal_batch_count + 1) * batch_size]
        batch_data = self.test_normal_data[batch_idx]
        for i in range(batch_size):
            img = np.load(batch_data[i, 0])
            img = img[1:, :]
            batch_img[i, :, :] = img
        batch_img = np.expand_dims(batch_img, axis=3)
        batch_label = self.test_normal_data[batch_idx, 1].astype(np.float).astype(np.int)
        self.test_normal_batch_count += 1
        if self.test_normal_batch_count == max_batch_count:
            self.test_normal_batch_count == 0
        return batch_img, batch_label

    def test_next_abnormal_batch(self):
        batch_size = self.train_flags.batch_size
        max_batch_count = int(self.test_abnormal_data.shape[0] / batch_size)
        if self.test_abnormal_batch_count == 0:
            np.random.shuffle(self.test_abnormal_indices)
        batch_img = np.zeros(shape = (batch_size, 4, self.data_flags.Ls),
                             dtype = np.float32)
        batch_idx = self.test_abnormal_indices[self.test_abnormal_batch_count * batch_size:\
                                      (self.test_abnormal_batch_count + 1) * batch_size]
        batch_data = self.test_abnormal_data[batch_idx]
        for i in range(batch_size):
            img = np.load(batch_data[i, 0])
            img = img[1:, :]
            batch_img[i, :, :] = img
        batch_img = np.expand_dims(batch_img, axis=3)
        batch_label = self.test_abnormal_data[batch_idx, 1].astype(np.float).astype(np.int)
        self.test_abnormal_batch_count += 1
        if self.test_abnormal_batch_count == max_batch_count:
            self.test_abnormal_batch_count == 0
        return batch_img, batch_label

    def train_model(self):
        print('[i]  Function : train_model')
        self.dir_result = os.path.join(self.sys_flags.dir_data_base, 'result')
        self._make_dir(self.dir_result)
        max_epoch = self.train_flags.max_epoch
        dir_result = os.path.join(self.dir_result, 
                        'vgg16_lr_%f_l2_%f'%(self.train_flags.learning_rate, 
                                             self.train_flags.l2_regul))
        self._make_dir(dir_result)
        train_writer = tf.summary.create_file_writer(os.path.join(dir_result, 
                                                                 'train_summary'))
        test_writer = tf.summary.create_file_writer(os.path.join(dir_result,
                                                                 'test_summary'))
        while max_epoch > self.epoch:
            train_acc, loss_total = self.train_one_iter()
            with train_writer.as_default():
                tf.summary.scalar('accuracy', train_acc, step = self.iter)
                tf.summary.scalar('loss_total', loss_total, step = self.iter)
            if self.iter % 5 == 0:
                test_acc, test_loss_total, test_normal_acc, test_abnormal_acc = self.test_one_iter()
                with test_writer.as_default():
                    tf.summary.scalar('accuracy', test_acc, step = self.iter)
                    tf.summary.scalar('loss_total', test_loss_total, step = self.iter)
                    tf.summary.scalar('abnormal_acc', test_abnormal_acc, step = self.iter)
                    tf.summary.scalar('normal_acc', test_normal_acc, step = self.iter)


    def train_one_iter(self):
        batch_img, batch_label = self.train_next_batch()
        with tf.GradientTape() as tape:
            logits = self.model(batch_img)
            softmax = tf.nn.softmax(logits)
            train_loss_value = self.loss_object(batch_label, logits)
            l2_weights = 0
            for i in range(len(self.model.weights)):
                l2_weights += tf.keras.regularizers.l2(self.train_flags.l2_regul)\
                                                      (self.model.weights[i])
            loss_total = train_loss_value + l2_weights
        train_acc = self.acc(batch_label, logits)
        self.acc.reset_states()
        grads = tape.gradient(loss_total, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.iter += 1
        return train_acc.numpy(), loss_total.numpy()

    def test_one_iter(self):
        batch_img, batch_label = self.test_next_batch()
        logits = self.model(batch_img)
        softmax = tf.nn.softmax(logits)
        test_loss_value = self.loss_object(batch_label, logits)
        for i in range(len(self.model.weights)):
            test_loss_value += tf.keras.regularizers.l2(self.train_flags.l2_regul)(self.model.weights[i])
        test_acc = self.acc(batch_label, logits)
        self.acc.reset_states()
        batch_img, batch_label = self.test_next_normal_batch()
        logits_normal = self.model(batch_img)
        softmax_normal = tf.nn.softmax(logits_normal)
        test_normal_acc = self.acc(batch_label, logits_normal)
        self.acc.reset_states()
        batch_img, batch_label = self.test_next_abnormal_batch()
        logits_abnormal = self.model(batch_img)
        softmax_abnormal = tf.nn.softmax(logits_abnormal)
        test_abnormal_acc = self.acc(batch_label, logits_abnormal)
        self.acc.reset_states()
        return test_acc.numpy(), test_loss_value.numpy(), \
                test_normal_acc.numpy(), test_abnormal_acc.numpy()

    def _make_dir(self, dir_):
        if not os.path.exists(dir_):
            os.mkdir(dir_)


if __name__ == '__main__':
    np.random.seed(sys_flags.random_seed)
    tf.random.set_seed(sys_flags.random_seed)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.experimental.set_memory_growth(gpus[1], True)
            tf.config.experimental.set_memory_growth(gpus[2], True)
            tf.config.experimental.set_memory_growth(gpus[3], True)
            tf.config.experimental.set_visible_devices(gpus[train_flags.gpu_num], 'GPU')
        except RuntimeError as e:
            print(e)
    train_class()
