import numpy as np
from torch.utils.data.dataset import Dataset


class trainloader_class(Dataset):
    def __init__(self, factory_module, sys_flags, data_flags, train_flags):
        super(trainloader_class, self).__init__()
        factory_class = factory_module.factory_class(sys_flags, data_flags, train_flags)
        dir_train_data = factory_class.dir_train_data
        train_data = np.load(dir_train_data)
        self.dirs = train_data[:, 0].tolist()
        self.labels = (train_data[:, 1].astype(np.float).astype(np.int)).tolist()

    def __getitem__(self, idx):
        item_dir = self.dirs[idx]
        item_label = int(self.labels[idx])
        img = np.load(item_dir)
        img = np.expand_dims(img, axis=0)
        return img, item_label

    def __len__(self):
        return len(self.dirs)

class testloader_class(Dataset):
    def __init__(self, factory_module, sys_flags, data_flags, train_flags):
        super(testloader_class, self).__init__()
        factory_class = factory_module.factory_class(sys_flags, data_flags, train_flags)
        dir_test_data = factory_class.dir_test_data
        test_data = np.load(dir_test_data)
        self.dirs = test_data[:, 0].tolist()
        self.labels = (test_data[:, 1].astype(np.float).astype(np.int)).tolist()

    def __getitem__(self, idx):
        item_dir = self.dirs[idx]
        item_label = int(self.labels[idx])
        img = np.load(item_dir)
        img = np.expand_dims(img, axis=0)
        return img, item_label

    def __len__(self):
        return len(self.dirs)

class train_normal_class(Dataset):
    def __init__(self, factory_module, sys_flags, data_flags, train_flags):
        super(train_normal_class, self).__init__()
        factory_class = factory_module.factory_class(sys_flags, data_flags, train_flags)
        dir_train_data = factory_class.dir_train_data
        train_data = np.load(dir_train_data)
        normal_indices = np.where(train_data[:, 1] == '0.0')[0]
        train_data = train_data[normal_indices]
        self.dirs = train_data[:, 0].tolist()
        self.labels = (train_data[:, 1].astype(np.float).astype(np.int)).tolist()

    def __getitem__(self, idx):
        item_dir = self.dirs[idx]
        item_label = int(self.labels[idx])
        img = np.load(item_dir)
        img = np.expand_dims(img, axis=0)
        return img, item_label

    def __len__(self):
        return len(self.dirs)

class train_abnormal_class(Dataset):
    def __init__(self, factory_module, sys_flags, data_flags, train_flags):
        super(train_abnormal_class, self).__init__()
        factory_class = factory_module.factory_class(sys_flags, data_flags, train_flags)
        dir_train_data = factory_class.dir_train_data
        train_data = np.load(dir_train_data)
        abnormal_indices = np.where(train_data[:, 1] == '1.0')[0]
        train_data = train_data[abnormal_indices]
        self.dirs = train_data[:, 0].tolist()
        self.labels = (train_data[:, 1].astype(np.float).astype(np.int)).tolist()

    def __getitem__(self, idx):
        item_dir = self.dirs[idx]
        item_label = int(self.labels[idx])
        img = np.load(item_dir)
        img = np.expand_dims(img, axis=0)
        return img, item_label

    def __len__(self):
        return len(self.dirs)

class test_normal_class(Dataset):
    def __init__(self, factory_module, sys_flags, data_flags, train_flags):
        super(test_normal_class, self).__init__()
        factory_class = factory_module.factory_class(sys_flags, data_flags, train_flags)
        dir_test_data = factory_class.dir_test_data
        test_data = np.load(dir_test_data)
        normal_indices = np.where(test_data[:, 1] == '0.0')[0]
        test_data = test_data[normal_indices]
        self.dirs = test_data[:, 0].tolist()
        self.labels = (test_data[:, 1].astype(np.float).astype(np.int)).tolist()

    def __getitem__(self, idx):
        item_dir = self.dirs[idx]
        item_label = int(self.labels[idx])
        img = np.load(item_dir)
        img = np.expand_dims(img, axis=0)
        return img, item_label

    def __len__(self):
        return len(self.dirs)

class test_abnormal_class(Dataset):
    def __init__(self, factory_module, sys_flags, data_flags, train_flags):
        super(test_abnormal_class, self).__init__()
        factory_class = factory_module.factory_class(sys_flags, data_flags, train_flags)
        dir_test_data = factory_class.dir_test_data
        test_data = np.load(dir_test_data)
        abnormal_indices = np.where(test_data[:, 1] == '1.0')[0]
        test_data = test_data[abnormal_indices]
        self.dirs = test_data[:, 0].tolist()
        self.labels = (test_data[:, 1].astype(np.float).astype(np.int)).tolist()

    def __getitem__(self, idx):
        item_dir = self.dirs[idx]
        item_label = int(self.labels[idx])
        img = np.load(item_dir)
        img = np.expand_dims(img, axis=0)
        return img, item_label

    def __len__(self):
        return len(self.dirs)
