import argparse
import os

def get_sys_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_cwd', default = os.getcwd())
    parser.add_argument('--dir_data_base', default = os.path.join('/mnt', 'mnt', 'sdd', 'Plasma'))
    parser.add_argument('--dir_processed_data', default = os.path.join('/mnt', 'mnt', 'sdd',
                                                                       'Plasma', 'processed'))
    parser.add_argument('--module_factory', default = 'utils.factory')
    parser.add_argument('--random_seed', default=777)
    flags = parser.parse_args()
    return flags

def get_data_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Ls', default = 256)
    parser.add_argument('--Ls_shift', default = 10)
    parser.add_argument('--use_normalize', default = True)
    parser.add_argument('--train_data_ratio', default = 0.8)
    flags = parser.parse_args()
    return flags

def get_train_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default = 0.001)
    parser.add_argument('--batch_size', default = 256)
    flags = parser.parse_args()
    return flags