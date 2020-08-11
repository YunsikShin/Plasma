import numpy as np
import csv
import os
import pdb
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

class factory_class:
    def __init__(self, sys_flags, data_flags):
        print('[i]  Class : factory_class')
        self.sys_flags = sys_flags
        self.data_flags = data_flags
        self.prepare_dirs()
        self.txt_to_npy()
        self.detect_arcing()
        pdb.set_trace()
        self.make_npy_figs()
        self.rm_after_arching()
        self.get_statistical_value()
        self.get_img()
        self.make_train_data()

    def prepare_dirs(self):
        print('[i]  Function : prepare_dirs')
        sys_flags = self.sys_flags
        data_flags = self.data_flags
        data_flags_dict = vars(data_flags)
        dir_processed_data = sys_flags.dir_processed_data
        self._make_dir(dir_processed_data)
        env_list = os.listdir(dir_processed_data)
        if not env_list:
            dir_processed_data = os.path.join(dir_processed_data, 'env_1')
            self._make_dir(dir_processed_data)
            dir_env_pkl = os.path.join(dir_processed_data, 'data_flags.pkl')
            with open(dir_env_pkl, 'wb') as f:
                pickle.dump(data_flags_dict, f, pickle.HIGHEST_PROTOCOL)
        else:
            find_same_data_flag = False
            for env_name in env_list:
                dir_env_pkl = os.path.join(dir_processed_data, env_name, 'data_flags.pkl')
                with open(dir_env_pkl, 'rb') as f:
                    saved_pkl = pickle.load(f)
                if len(data_flags_dict) != len(saved_pkl):
                    pass
                elif data_flags_dict == saved_pkl:
                    dir_processed_data = os.path.join(dir_processed_data, env_name)
                    find_same_data_flag = True
                    break
            if find_same_data_flag == False:
                new_env_name = 'env_%d'%(len(env_list)+1)
                dir_processed_data = os.path.join(dir_processed_data, new_env_name)
                self._make_dir(dir_processed_data)
                with open(os.path.join(dir_processed_data, 'data_flags.pkl'), 'wb') as f:
                    pickle.dump(data_flags_dict, f, pickle.HIGHEST_PROTOCOL)
        self.dir_processed_data = dir_processed_data

    def txt_to_npy(self):
        print('[Shin]  Function : txt_to_npy')
        self.dir_npys = os.path.join(self.dir_processed_data, 'txt_to_npy')
        if not os.path.exists(self.dir_npys):
            self._make_dir(self.dir_npys)
            dir_raw_data = os.path.join(self.sys_flags.dir_data_base, 'raw_data')
            data_list = os.listdir(dir_raw_data)
            for dir_idx in range(len(data_list)):
                print('[Shin]    Processing %s data'%data_list[dir_idx])
                dir_data = os.path.join(dir_raw_data, data_list[dir_idx])
                f = open(dir_data, 'r')
                num_line = 0
                total_data = []
                while True:
                    line = f.readline()
                    if not line:break
                    line_split = line.split('  ')
                    total_data.append(line_split[1:])
                total_data = np.array(total_data).astype(np.float32)
                total_data = np.transpose(total_data)
                dir_npy = os.path.join(self.dir_npys, data_list[dir_idx])
                np.save(dir_npy, total_data)

    def detect_arcing(self):
        print('[i]  Function : detect_arcing')
        scen_length = 10000
        list_npys = os.listdir(self.dir_npys)
        for npy in list_npys:
            data = np.load(os.path.join(self.dir_npys, npy))
            data = np.transpose(data)
            num_scen = int(data.shape[1] / scen_length)
            avg_d = np.zeros(shape = (4, num_scen))
            for scen_idx in range(num_scen):
                time = data[0, scen_length*scen_idx : scen_length*(scen_idx+1)]
                data_1 = data[1, scen_length*scen_idx : scen_length*(scen_idx+1)]
                data_2 = data[2, scen_length*scen_idx : scen_length*(scen_idx+1)]
                data_3 = data[3, scen_length*scen_idx : scen_length*(scen_idx+1)]
                data_4 = data[4, scen_length*scen_idx : scen_length*(scen_idx+1)]
                avg_d[0, scen_idx] = np.mean(data_1)
                avg_d[1, scen_idx] = np.mean(data_2)
                avg_d[2, scen_idx] = np.mean(data_3)
                avg_d[3, scen_idx] = np.mean(data_4)
            idx_upper = np.where(avg_d[2, :] < -65)[0]
            idx_lower = np.where(avg_d[2, :] > -67)[0]
            indices = []
            for idx in range(num_scen):
                if (idx in idx_upper) and (idx in idx_lower):
                    indices.append(idx)
            print(npy, len(indices))
        pdb.set_trace()

    def make_npy_figs(self):
        print('[i]    Function : make_npy_figs')
        num_signals = 4
        list_data = os.listdir(self.dir_npys)
        num_data = len(list_data)
        for npy_idx in range(num_data):
            data = np.load(os.path.join(self.dir_npys, list_data[npy_idx]))
            data = np.transpose(data)
            for sig_idx in range(num_signals):
                signal = data[sig_idx, :]
                dir_fig = os.path.join(os.getcwd(), 
                                       '%s_signal_%d.png'%(list_data[npy_idx], sig_idx))
                plt.figure(figsize=(20,20))
                plt.plot(signal[:50000])
                plt.grid()
                plt.savefig(dir_fig)
                plt.close()
        pdb.set_trace()

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

