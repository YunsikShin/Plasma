import numpy as np
import csv
import os
import pdb
import matplotlib.pyplot as plt

def _make_dir(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)


class factory_class:
    def __init__(self):
        self.initialization()
        self.csv_to_npy()
        self.make_npy_figs()

    def initialization(self):
        self.dir_cwd = os.getcwd()
        self.dir_raw_data_normal = os.path.join(self.dir_cwd, 'raw_data', 'normal.csv')
        self.dir_raw_data_abnormal = os.path.join(self.dir_cwd, 'raw_data', 'abnormal.csv')
        self.dir_processed = os.path.join(self.dir_cwd, 'processed')
        self._make_dir(self.dir_processed)

    def csv_to_npy(self):
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
                np.save(dir_seg, seg)
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
                np.save(dir_seg, seg)
            del data_abnormal

    def make_npy_figs(self):
        self.dir_npy_figs = os.path.join(self.dir_processed, 'npy_figs')
        #if not os.path.exists(self.dir_npy_figs):
        if True:
            self._make_dir(self.dir_npy_figs)
            num_normal = len(os.listdir(self.dir_processed_npy_normal))
            for npy_idx in range(num_normal):
                npy_data = np.load(os.path.join(self.dir_processed_npy_normal, '%d.npy'%npy_idx))
                for char_idx in range(4):
                    array_data = npy_data[:, char_idx+1]
                    dir_fig = os.path.join(self.dir_npy_figs, 'char_%d_npy_%d.png'%(char_idx+1, npy_idx))
                    plt.figure()
                    plt.plot(array_data)
                    plt.grid()
                    plt.savefig(dir_fig)
                    plt.close()
            num_abnormal = len(os.listdir(self.dir_processed_npy_abnormal))
            for npy_idx in range(num_abnormal):
                npy_data = np.load(os.path.join(self.dir_processed_npy_abnormal, '%d.npy')%npy_idx)
                for char_idx in range(4):
                    array_data = npy_data[:, char_idx+1]
                    dir_fig = os.path.join(self.dir_npy_figs, 'char_%d_npy_%d.png'%(char_idx+1, npy_idx))
                    plt.figure()
                    plt.plot(array_data)
                    plt.grid()
                    plt.savefig(dir_fig)
                    plt.close()






        

    def _make_dir(self, dir_):
        if not os.path.exists(dir_):
            os.mkdir(dir_)

factory_class()
