import numpy as np
import csv
import os
import pdb
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

class factory_class:
    def __init__(self, sys_flags, data_flags, train_flags):
        print('[Shin]  Class : factory_class')
        self.sys_flags = sys_flags
        self.data_flags = data_flags
        self.train_flags = train_flags
        self.sig_names = ['time stamp', 'Probe Voltage', 'AIP Current', 'AIP Voltage', 
                          'Floating-potential probe Voltage']
        self.prepare_dirs()
        self.csv_to_npy()
        self.split_normal_abnormal()
        self.data_analysis()
        self.get_statistical_value()
        self.get_img()
        self.make_train_data()

    def prepare_dirs(self):
        print('[Shin]  Function : prepare_dirs')
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

    def csv_to_npy(self):
        print('[Shin]  Function : txt_to_npy')
        self.dir_npys = os.path.join(self.dir_processed_data, 'csv_to_npy')
        if not os.path.exists(self.dir_npys):
            self._make_dir(self.dir_npys)
            dir_raw_data = os.path.join(self.sys_flags.dir_data_base, 'raw_data')
            data_names = os.listdir(dir_raw_data)
            for data_name in data_names:
                print('[Shin]    Processing %s data'%data_name)
                dir_data = os.path.join(dir_raw_data, data_name)
                data = pd.read_csv(dir_data)
                npy = np.transpose(data.to_numpy())
                npy_name = '_'.join(data_name.split('.')[0].split('_')[-2:])
                dir_npy = os.path.join(self.dir_npys, npy_name)
                np.save(dir_npy, npy)

    def split_normal_abnormal(self):
        print('[Shin]  Function : split_normal_abnormal')
        self.dir_split_normal = os.path.join(self.dir_processed_data, 'npy_split_normal')
        self.dir_split_abnormal = os.path.join(self.dir_processed_data, 'npy_split_abnormal')
        data_list = os.listdir(self.dir_npys)
        if not os.path.exists(self.dir_split_normal):
            os.mkdir(self.dir_split_normal)
            os.mkdir(self.dir_split_abnormal)
            for data_name in data_list:
                if data_name.split('.')[0].split('_')[0] == 'normal':
                    normal_abnormal = 'normal'
                elif data_name.split('.')[0].split('_')[0] == 'abnormal':
                    normal_abnormal = 'abnormal'
                dir_data = os.path.join(self.dir_npys, data_name)
                data = np.load(dir_data)
                t_length = data.shape[1]
                num_scen = int(t_length // 10000)
                for scen_idx in range(num_scen):
                    if normal_abnormal == 'normal':
                        num_scen_data = len(os.listdir(self.dir_split_normal))
                        dir_scen_data = os.path.join(self.dir_split_normal, 
                                                     'scen_%d.npy'%num_scen_data)
                    elif normal_abnormal == 'abnormal':
                        num_scen_data = len(os.listdir(self.dir_split_abnormal))
                        dir_scen_data = os.path.join(self.dir_split_abnormal, 
                                                     'scen_%d.npy'%num_scen_data)
                    scen_data = data[:, 10000*scen_idx : 10000*(scen_idx+1)]
                    np.save(dir_scen_data, scen_data)
        num_normal = len(os.listdir(self.dir_split_normal))
        num_abnormal = len(os.listdir(self.dir_split_abnormal))
        print('[Shin]    Num normal   :', num_normal)
        print('[Shin]    Num abnormal :', num_abnormal)

    def data_analysis(self):
        print('[Shin]  Function : data_analysis')
        dir_figs = os.path.join(os.getcwd(), 'data_analysis')
        if not os.path.exists(dir_figs):
            self._make_dir(dir_figs)
            normal_list = os.listdir(self.dir_split_normal)
            abnormal_list = os.listdir(self.dir_split_abnormal)
            for sig_idx in range(4):
                normal_mean_list = []
                normal_std_list = []
                abnormal_mean_list = []
                abnormal_std_list = []
                for normal_name in normal_list:
                    dir_data = os.path.join(self.dir_split_normal, normal_name)
                    data = np.load(dir_data)
                    sig_data = data[sig_idx+1, :6000]
                    sig_mean = np.mean(sig_data)
                    sig_std = np.std(sig_data)
                    normal_mean_list.append(sig_mean)
                    normal_std_list.append(sig_std)
                for abnormal_name in abnormal_list:
                    dir_data = os.path.join(self.dir_split_abnormal, abnormal_name)
                    data = np.load(dir_data)
                    sig_data = data[sig_idx+1, :6000]
                    sig_mean = np.mean(sig_data)
                    sig_std = np.std(sig_data)
                    abnormal_mean_list.append(sig_mean)
                    abnormal_std_list.append(sig_std)
                num_normal = len(normal_mean_list)
                num_abnormal = len(abnormal_mean_list)
                x = np.linspace(0, num_normal, num_abnormal)
                plt.figure(figsize=(25, 15))
                plt.plot(normal_mean_list, alpha=0.7, c='b')
                plt.plot(x, abnormal_mean_list, alpha=0.7, c='r')
                plt.grid()
                plt.savefig(os.path.join(dir_figs, self.sig_names[sig_idx+1]))

    def get_statistical_value(self):
        print('[Shin]    Function : get_statistical_value')
        self.dir_normal_mean = os.path.join(self.dir_processed_data, 'normal_mean.npy')
        self.dir_normal_std = os.path.join(self.dir_processed_data, 'normal_std.npy')
        if not os.path.exists(self.dir_normal_mean):
            normal_mean = []
            normal_std = []
            normal_list = os.listdir(self.dir_split_normal)
            for normal_name in normal_list:
                dir_data = os.path.join(self.dir_split_normal, normal_name)
                data = np.load(dir_data)[1:, :]
                mean = np.expand_dims(np.mean(data, axis=1), axis=1)
                std = np.expand_dims(np.std(data, axis=1), axis=1)
                if len(normal_mean) == 0:
                    normal_mean = mean
                    normal_std = std
                else:
                    normal_mean = np.append(normal_mean, mean, axis=1)
                    normal_std = np.append(normal_std, std, axis=1)
            normal_mean = np.mean(normal_mean, axis=1)
            normal_std = np.mean(normal_std, axis=1)
            np.save(self.dir_normal_mean, normal_mean)
            np.save(self.dir_normal_std, normal_std)

    def get_img(self):
        print('[i]    Function : get_img')
        data_flags = self.data_flags
        self.dir_img = os.path.join(self.dir_processed_data, 'img')
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
            num_normal = len(os.listdir(self.dir_split_normal))
            num_abnormal = len(os.listdir(self.dir_split_abnormal))
            num_normal_train = int(num_normal * data_flags.train_data_ratio)
            num_abnormal_train = int(num_abnormal * data_flags.train_data_ratio)
            normal_list = os.listdir(self.dir_split_normal)
            abnormal_list = os.listdir(self.dir_split_abnormal)
            np.random.shuffle(normal_list)
            np.random.shuffle(abnormal_list)
            normal_train = normal_list[:num_normal_train]
            normal_test = normal_list[num_normal_train:]
            abnormal_train = abnormal_list[:num_abnormal_train]
            abnormal_test = abnormal_list[num_abnormal_train:]
            for normal_name in normal_list:
                if normal_name in normal_train:
                    dir_img = self.dir_img_normal_train
                elif normal_name in normal_test:
                    dir_img = self.dir_img_normal_test
                dir_data = os.path.join(self.dir_split_normal, normal_name)
                data = np.load(dir_data)
                num_saved_img = len(os.listdir(dir_img))
                npy_name = '%d.npy'%num_saved_img
                dir_img = os.path.join(dir_img, npy_name)
                mean = np.expand_dims(np.load(self.dir_normal_mean), axis=1)
                std = np.expand_dims(np.load(self.dir_normal_std), axis=1)
                mean_data = np.divide(data[1:,:]  - mean, std)
                np.save(dir_img, mean_data[:, :data_flags.end_point])
            for abnormal_name in abnormal_list:
                if abnormal_name in abnormal_train:
                    dir_img = self.dir_img_abnormal_train
                elif abnormal_name in abnormal_test:
                    dir_img = self.dir_img_abnormal_test
                dir_data = os.path.join(self.dir_split_abnormal, abnormal_name)
                data = np.load(dir_data)
                num_saved_img = len(os.listdir(dir_img))
                npy_name = '%d.npy'%num_saved_img
                dir_img = os.path.join(dir_img, npy_name)
                mean = np.expand_dims(np.load(self.dir_normal_mean), axis=1)
                std = np.expand_dims(np.load(self.dir_normal_std), axis=1)
                mean_data = np.divide(data[1:, :] - mean, std)
                np.save(dir_img, mean_data[:, :data_flags.end_point])
        print('[Shin]    Num normal train   :', len(os.listdir(self.dir_img_normal_train)))
        print('[Shin]    Num normal test    :', len(os.listdir(self.dir_img_normal_test)))
        print('[Shin]    Num abnormal train :', len(os.listdir(self.dir_img_abnormal_train)))
        print('[Shin]    Num abnormal test  :', len(os.listdir(self.dir_img_abnormal_test)))
                    
    def make_train_data(self):
        print('[i]    Function : make_train_data')
        self.dir_train_data = os.path.join(self.dir_processed_data, 
                                           'train_method_%s.npy'%(self.train_flags.train_method))
        self.dir_test_data = os.path.join(self.dir_processed_data,
                                          'test_method_%s.npy'%(self.train_flags.train_method))
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

    #def detect_arcing(self):
    #    print('[Shin]  Function : detect_arcing')
    #    scen_length = 10000
    #    list_npys = np.sort(os.listdir(self.dir_npys))
    #    total_avg_d2 = []
    #    for npy in list_npys:
    #        data = np.load(os.path.join(self.dir_npys, npy))
    #        num_scen = int(data.shape[1] / scen_length)
    #        avg_d = np.zeros(shape = (4, num_scen))
    #        for scen_idx in range(num_scen):
    #            time = data[0, scen_length*scen_idx : scen_length*(scen_idx+1)]
    #            data_1 = data[1, scen_length*scen_idx : scen_length*(scen_idx+1)]
    #            data_2 = data[2, scen_length*scen_idx : scen_length*(scen_idx+1)]
    #            data_3 = data[3, scen_length*scen_idx : scen_length*(scen_idx+1)]
    #            data_4 = data[4, scen_length*scen_idx : scen_length*(scen_idx+1)]
    #            avg_d[0, scen_idx] = np.mean(data_1)
    #            avg_d[1, scen_idx] = np.mean(data_2)
    #            avg_d[2, scen_idx] = np.mean(data_3)
    #            avg_d[3, scen_idx] = np.mean(data_4)
    #        idx_upper = np.where(avg_d[2, :] < -65)[0]
    #        idx_lower = np.where(avg_d[2, :] > -67)[0]
    #        indices = []
    #        for idx in range(num_scen):
    #            if (idx in idx_upper) and (idx in idx_lower):
    #                indices.append(idx)
    #        print(npy, len(indices))
    #        total_avg_d2.extend(avg_d[2, :])
    #    d2_len = len(total_avg_d2)
    #    upper = [-65]*d2_len
    #    lower = [-68]*d2_len
    #    dir_fig = os.path.join(os.getcwd(), 'avg_d.png')
    #    plt.figure()
    #    plt.plot(total_avg_d2)
    #    plt.grid()
    #    plt.plot(upper, c='r')
    #    plt.plot(lower, c='r')
    #    plt.savefig(dir_fig)
    #    plt.close()
    #    pdb.set_trace()
