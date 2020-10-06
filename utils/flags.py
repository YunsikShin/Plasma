import argparse
import os


def get_sys_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_cwd', default=os.getcwd())
    parser.add_argument('--dir_data_base', default=os.path.join('/mnt', 'mnt',
                                                                'sdd', 'Plasma'))
    parser.add_argument('--dir_processed_data', default=os.path.join('/mnt', 'mnt', 'sdd',
                                                                     'Plasma', 'processed'))
    parser.add_argument('--module_factory', default='utils.factory')
    parser.add_argument('--random_seed', default=777)
    flags = parser.parse_args()
    return flags

def get_data_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_normalize', default=True)
    parser.add_argument('--train_data_ratio', default=0.8)
    parser.add_argument('--end_point', default=6000)
    flags = parser.parse_args()
    return flags

def get_train_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_num', default=1)
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--SGD_momentum', default=0.9)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--max_epoch', default=1000)
    parser.add_argument('--l2_regul', default=0.001)
    parser.add_argument('--model_name', default='vgg16')

    parser.add_argument('--train_method', default='classification')
    flags = parser.parse_args()
    return flags
