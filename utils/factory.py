import numpy as np
import csv
import os
import pdb
import matplotlib.pyplot as plt

class factory_class:
    def __init__(self, sys_flags, data_flags):
        print('[i]  Class : factory_class')
        self.sys_flags = sys_flags
        self.data_flags = data_flags
        self.initialization()
        self.csv_to_npy()
        self.make_npy_figs()
        self.rm_after_arching()
        self.get_statistical_value()
        self.get_img()
        self.make_train_data()

    def initialization(self):
        sys_flags = self.sys_flags
        self.dir_cwd = os.getcwd()
        self.dir_raw_data_normal = os.path.join(sys_flags.dir_data_base, 'raw_data', 'normal.csv')
        self.dir_raw_data_abnormal = os.path.join(sys_flags.dir_data_base, 'raw_data', 'abnormal.csv')
        self.dir_processed = os.path.join(sys_flags.dir_data_base, 'processed')
        self._make_dir(self.dir_processed)

    def csv_to_npy(self):
        print('[i]    Function : csv_to_npy')
        dir_processed_npy = os.path.join(self.dir_processed, 'npy')
        self.dir_processed_npy_normal = os.path.join(dir_processed_npy, 'normal')
        self.dir_processed_npy_abnormal = os.path.join(dir_processed_npy, 'abnormal')
        if not os.path.exists(dir_processed_npy):
            self._make_dir(dir_processed_npy)
            self._make_dir(self.dir_processed_npy_normal)
            with open(self.dir_raw_data_normal, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                data_normal = list(reader)
                row_count_normal = len(data_normal)
            data_normal = np.array(data_normal)
            for seg_idx in range(int(row_count_normal/10000)):
                dir_seg = os.path.join(self.dir_processed_npy_normal, '%d.npy'%seg_idx)
                seg = data_normal[10000*seg_idx:10000*(seg_idx+1)]
                np.save(dir_seg, np.transpose(seg))
            del data_normal
            self._make_dir(self.dir_processed_npy_abnormal)
            with open(self.dir_raw_data_abnormal, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',', quotechar='|')
                data_abnormal = list(reader)
                row_count_abnormal = len(data_abnormal)
            data_abnormal = np.array(data_abnormal)
            for seg_idx in range(int(row_count_abnormal/10000)):
                dir_seg = os.path.join(self.dir_processed_npy_abnormal, '%d.npy'%seg_idx)
                seg = data_abnormal[10000*seg_idx:10000*(seg_idx+1)]
                np.save(dir_seg, np.transpose(seg))
            del data_abnormal
        self.num_npy_normal = len(os.listdir(self.dir_processed_npy_normal))
        self.num_npy_abnormal = len(os.listdir(self.dir_processed_npy_abnormal))

    def make_npy_figs(self):
        print('[i]    Function : make_npy_figs')
        self.dir_npy_figs = os.path.join(self.dir_processed, 'npy_figs')
        num_fig = 5
        if not os.path.exists(self.dir_npy_figs):
            self._make_dir(self.dir_npy_figs)
            for npy_idx in range(num_fig):
                npy_data_normal = np.load(os.path.join(self.dir_processed_npy_normal, 
                                                       '%d.npy'%npy_idx))
                npy_data_abnormal = np.load(os.path.join(self.dir_processed_npy_abnormal,
                                                         '%d.npy'%npy_idx))
                for char_idx in range(4):
                    arr_data_normal = npy_data_normal[char_idx+1, :].astype(np.float32)
                    arr_data_abnormal = npy_data_abnormal[char_idx+1, :].astype(np.float32)
                    dir_fig = os.path.join(self.dir_npy_figs, 
                                           'compare_signal_%d_npy_%d.png'%(char_idx+1, npy_idx))
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(arr_data_normal, alpha=0.5, lw=3, color='b')
                    ax.plot(arr_data_abnormal, alpha=0.5, lw=3, color='r')
                    plt.grid()
                    plt.savefig(dir_fig)
            #for npy_idx in range(num_fig):
            #    npy_data = np.load(os.path.join(self.dir_processed_npy_normal, '%d.npy'%npy_idx))
            #    for char_idx in range(4):
            #        array_data = npy_data[char_idx+1, :]
            #        dir_fig = os.path.join(self.dir_npy_figs, 
            #                               'normal_char_%d_npy_%d.png'%(char_idx+1, npy_idx))
            #        plt.figure()
            #        plt.plot(array_data)
            #        plt.grid()
            #        plt.savefig(dir_fig)
            #        plt.close()
            #num_abnormal = len(os.listdir(self.dir_processed_npy_abnormal))
            #for npy_idx in range(num_fig):
            #    npy_data = np.load(os.path.join(self.dir_processed_npy_abnormal, '%d.npy')%npy_idx)
            #    for char_idx in range(4):
            #        array_data = npy_data[char_idx+1, :]
            #        dir_fig = os.path.join(self.dir_npy_figs, 
            #                               'abnormal_char_%d_npy_%d.png'%(char_idx+1, npy_idx))
            #        plt.figure()
            #        plt.plot(array_data)
            #        plt.grid()
            #        plt.savefig(dir_fig)
            #        plt.close()

    def rm_after_arching(self):
        print('[i]    Function : rm_after_arching')
        self.dir_npy_abnormal_rm = os.path.join(self.dir_processed, 'npy', 'rm_abnormal')
        if not os.path.exists(self.dir_npy_abnormal_rm):
            self._make_dir(self.dir_npy_abnormal_rm)
            for npy_idx in range(self.num_npy_abnormal):
                npy_data = np.load(os.path.join(self.dir_processed_npy_abnormal, '%d.npy'%npy_idx))
                npy_useful = npy_data[:, :7000]
                np.save(os.path.join(self.dir_npy_abnormal_rm, '%d.npy'%npy_idx), npy_useful)

    def get_statistical_value(self):
        print('[i]    Function : get_statistical_value')
        self.dir_normal_mean = os.path.join(self.dir_processed, 'normal_mean.npy')
        self.dir_normal_std = os.path.join(self.dir_processed, 'normal_std.npy')
        self.dir_abnormal_mean = os.path.join(self.dir_processed, 'abnormal_mean.npy')
        self.dir_abnormal_std = os.path.join(self.dir_processed, 'abnormal_std.npy')
        if not os.path.exists(self.dir_normal_mean):
            normal_mean = 0
            normal_std = 0
            for npy_idx in range(self.num_npy_normal):
                npy_data = np.load(os.path.join(self.dir_processed_npy_normal, '%d.npy'%npy_idx))
                npy_data = npy_data.astype(np.float32)
                normal_mean += np.mean(npy_data, axis=1)
                normal_std += np.std(npy_data, axis=1)
            normal_mean /= self.num_npy_normal
            normal_std /= self.num_npy_normal
            np.save(self.dir_normal_mean, normal_mean)
            np.save(self.dir_normal_std, normal_std)

    def get_img(self):
        print('[i]    Function : get_img')
        data_flags = self.data_flags
        self.dir_img = os.path.join(self.dir_processed, 'img_Ls_%d'%(data_flags.Ls))
        self.dir_img_normal_train = os.path.join(self.dir_img, 'normal_train')
        self.dir_img_normal_test = os.path.join(self.dir_img, 'normal_test')
        self.dir_img_abnormal_train = os.path.join(self.dir_img, 'abnormal_train')
        self.dir_img_abnormal_test = os.path.join(self.dir_img, 'abnormal_test')
        if not os.path.exists(self.dir_img):
            self._make_dir(self.dir_img)
            self._make_dir(self.dir_img_normal_train)
            self._make_dir(self.dir_img_normal_test)
            self._make_dir(self.dir_img_abnormal_train)
            self._make_dir(self.dir_img_abnormal_test)
            num_normal_train = int(self.num_npy_normal * self.data_flags.train_data_ratio) 
            num_abnormal_train = int(self.num_npy_abnormal * self.data_flags.train_data_ratio)
            normal_npy_list = os.listdir(self.dir_processed_npy_normal)
            abnormal_npy_list = os.listdir(self.dir_npy_abnormal_rm)
            np.random.shuffle(normal_npy_list)
            np.random.shuffle(abnormal_npy_list)
            Ls = data_flags.Ls
            Ls_shift = data_flags.Ls_shift
            num_seg = int((10000 - Ls) / Ls_shift - 1)
            for npy_idx in range(num_normal_train):
                npy_data = np.load(os.path.join(self.dir_processed_npy_normal, 
                                                '%s'%normal_npy_list[npy_idx]))
                npy_data = npy_data.astype(np.float32)
                if data_flags.use_normalize:
                    mean = np.load(os.path.join(self.dir_processed, 'normal_mean.npy'))
                    std = np.load(os.path.join(self.dir_processed, 'normal_std.npy'))
                    mean = np.expand_dims(mean, axis=1)
                    std = np.expand_dims(std, axis=1)
                    npy_data = np.divide(npy_data - mean, std)
                num_saved_img = len(os.listdir(self.dir_img_normal_train))
                for seg_idx in range(num_seg):
                    dir_seg = os.path.join(self.dir_img_normal_train, '%d.npy'%(seg_idx 
                                                                                + num_saved_img))
                    seg = npy_data[:, Ls_shift*seg_idx : Ls_shift*seg_idx + Ls]
                    np.save(dir_seg, seg)
            for npy_idx in range(num_normal_train, self.num_npy_normal):
                npy_data = np.load(os.path.join(self.dir_processed_npy_normal, 
                                                '%s'%normal_npy_list[npy_idx]))
                npy_data = npy_data.astype(np.float32)
                if data_flags.use_normalize:
                    mean = np.load(os.path.join(self.dir_processed, 'normal_mean.npy'))
                    std = np.load(os.path.join(self.dir_processed, 'normal_std.npy'))
                    mean = np.expand_dims(mean, axis=1)
                    std = np.expand_dims(std, axis=1)
                num_saved_img = len(os.listdir(self.dir_img_normal_test))
                for seg_idx in range(num_seg):
                    dir_seg = os.path.join(self.dir_img_normal_test, '%d.npy'%(seg_idx
                                                                               +num_saved_img))
                    seg = npy_data[:, Ls_shift*seg_idx : Ls_shift*seg_idx + Ls]
                    np.save(dir_seg, seg)
            num_seg = int((7000 - Ls) / Ls_shift - 1)
            for npy_idx in range(num_abnormal_train):
                npy_data = np.load(os.path.join(self.dir_npy_abnormal_rm, 
                                                '%s'%abnormal_npy_list[npy_idx]))
                npy_data = npy_data.astype(np.float32)
                if data_flags.use_normalize:
                    mean = np.load(os.path.join(self.dir_processed, 'normal_mean.npy'))
                    std = np.load(os.path.join(self.dir_processed, 'normal_std.npy'))
                    mean = np.expand_dims(mean, axis=1)
                    std = np.expand_dims(std, axis=1)
                    npy_data = np.divide(npy_data - mean, std)
                num_saved_img = len(os.listdir(self.dir_img_abnormal_train))
                for seg_idx in range(num_seg):
                    dir_seg = os.path.join(self.dir_img_abnormal_train, '%d.npy'%(seg_idx
                                                                                  +num_saved_img))
                    seg = npy_data[:, Ls_shift*seg_idx : Ls_shift*seg_idx + Ls]
                    np.save(dir_seg, seg)
            for npy_idx in range(num_abnormal_train, self.num_npy_abnormal):
                npy_data = np.load(os.path.join(self.dir_npy_abnormal_rm, 
                                                '%s'%abnormal_npy_list[npy_idx]))
                npy_data = npy_data.astype(np.float32)
                if data_flags.use_normalize:
                    mean = np.load(os.path.join(self.dir_processed, 'normal_mean.npy'))
                    std = np.load(os.path.join(self.dir_processed, 'normal_std.npy'))
                    mean = np.expand_dims(mean, axis=1)
                    std = np.expand_dims(std, axis=1)
                    npy_data = np.divide(npy_data - mean, std)
                num_saved_img = len(os.listdir(self.dir_img_abnormal_test))
                for seg_idx in range(num_seg):
                    dir_seg = os.path.join(self.dir_img_abnormal_test, '%d.npy'%(seg_idx
                                                                                 +num_saved_img))
                    seg = npy_data[:, Ls_shift*seg_idx : Ls_shift*seg_idx + Ls]
                    np.save(dir_seg, seg)

    def make_train_data(self):
        print('[i]    Function : make_train_data')
        self.dir_train_data = os.path.join(self.dir_processed, 'train_data_Ls_%d.npy'\
                                                                %self.data_flags.Ls)
        self.dir_test_data = os.path.join(self.dir_processed, 'test_data_Ls_%d.npy'\
                                                                %self.data_flags.Ls)
        if (not os.path.exists(self.dir_train_data)) or (not os.path.exists(self.dir_test_data)):
            num_normal_train = len(os.listdir(self.dir_img_normal_train))
            num_normal_test = len(os.listdir(self.dir_img_normal_test))
            num_abnormal_train = len(os.listdir(self.dir_img_abnormal_train))
            num_abnormal_test = len(os.listdir(self.dir_img_abnormal_test))
            dirs_normal_train = []
            dirs_abnormal_train = []
            for idx in range(num_normal_train):
                dirs_normal_train.append(os.path.join(self.dir_img_normal_train, '%d.npy'%idx))
            for idx in range(num_abnormal_train):
                dirs_abnormal_train.append(os.path.join(self.dir_img_abnormal_train, '%d.npy'%idx))
            dirs_normal_train = np.expand_dims(np.array(dirs_normal_train), axis=1)
            dirs_abnormal_train = np.expand_dims(np.array(dirs_abnormal_train), axis=1)
            label_normal_train = np.expand_dims(np.zeros(shape=num_normal_train), axis=1)
            label_abnormal_train = np.expand_dims(np.ones(shape=num_abnormal_train), axis=1)
            normal_train_data = np.append(dirs_normal_train, label_normal_train, axis=1)
            abnormal_train_data = np.append(dirs_abnormal_train, label_abnormal_train, axis=1)
            train_data = np.append(normal_train_data, abnormal_train_data, axis=0)
            dirs_normal_test = []
            dirs_abnormal_test = []
            for idx in range(num_normal_test):
                dirs_normal_test.append(os.path.join(self.dir_img_normal_test, '%d.npy'%idx))
            for idx in range(num_abnormal_test):
                dirs_abnormal_test.append(os.path.join(self.dir_img_abnormal_test, '%d.npy'%idx))
            dirs_normal_test = np.expand_dims(np.array(dirs_normal_test), axis=1)
            dirs_abnormal_test = np.expand_dims(np.array(dirs_abnormal_test), axis=1)
            label_normal_test = np.expand_dims(np.zeros(shape=num_normal_test), axis=1)
            label_abnormal_test = np.expand_dims(np.ones(shape=num_abnormal_test), axis=1)
            normal_test_data = np.append(dirs_normal_test, label_normal_test, axis=1)
            abnormal_test_data = np.append(dirs_abnormal_test, label_abnormal_test, axis=1)
            test_data = np.append(normal_test_data, abnormal_test_data, axis=0)
            np.save(self.dir_train_data, train_data)
            np.save(self.dir_test_data, test_data)

    def _make_dir(self, dir_):
        if not os.path.exists(dir_):
            os.mkdir(dir_)

