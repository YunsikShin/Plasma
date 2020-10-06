import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
import utils.flags as flags_module
import utils.factory as factory_module
import numpy as np
import pdb
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Dataset

from dataloader.loaders import trainloader_class, testloader_class, train_normal_class, \
                               train_abnormal_class, test_normal_class, test_abnormal_class

class conv4_fc2(nn.Module):
    def __init__(self, flatten_size):
        super(conv4_fc2, self).__init__()
        self.flatten_size = flatten_size
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(4, 3), stride=(1,2))
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(1, 3), stride=(1,2))
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 3), stride=(1,2))
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(1, 3), stride=(1,2))
        self.pool = nn.MaxPool2d(1, 2)
        self.fc1 = nn.Linear(self.flatten_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class train_class:
    def __init__(self):
        ### Get Flags
        print('[Shin]  Class : train_class')
        self.sys_flags = flags_module.get_sys_flags()
        self.data_flags = flags_module.get_data_flags()
        self.train_flags = flags_module.get_train_flags()
        self.device = torch.device('cuda:%d'%self.train_flags.gpu_num)
        self.get_loaders()
        self.exp_name = 'Exp_lr_%.6f_momentum_%.2f_method_%s'%(self.train_flags.learning_rate,
                                                               self.train_flags.SGD_momentum,
                                                               self.train_flags.train_method)
        self.dir_result = os.path.join('/mnt', 'mnt', 'sdc', 'ysshin', 'Plasma', self.exp_name)
        self._make_dir(self.dir_result)
        self.net = conv4_fc2(5952).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.train_flags.learning_rate,
                                   momentum = self.train_flags.SGD_momentum)
        self.train_model()

    def get_loaders(self):
        train_loader = trainloader_class(factory_module, self.sys_flags, self.data_flags, 
                                         self.train_flags)
        test_loader = testloader_class(factory_module, self.sys_flags, self.data_flags, 
                                       self.train_flags)
        train_normal_loader = train_normal_class(factory_module, self.sys_flags, self.data_flags,
                                                 self.train_flags)
        train_abnormal_loader = train_abnormal_class(factory_module, self.sys_flags, 
                                                     self.data_flags, self.train_flags)
        test_normal_loader = test_normal_class(factory_module, self.sys_flags, self.data_flags,
                                               self.train_flags)
        test_abnormal_loader = test_abnormal_class(factory_module, self.sys_flags, self.data_flags,
                                                   self.train_flags)
        self.train_loader = torch.utils.data.DataLoader(train_loader, 
                                                        batch_size = self.train_flags.batch_size,
                                                        shuffle=True, num_workers = 8)
        self.test_loader = torch.utils.data.DataLoader(test_loader,
                                                       batch_size = self.train_flags.batch_size,
                                                       shuffle=True, num_workers = 8)
        self.train_normal_loader = torch.utils.data.DataLoader(train_normal_loader, 
                                                        batch_size = self.train_flags.batch_size,
                                                        shuffle=True, num_workers = 8)
        self.train_abnormal_loader = torch.utils.data.DataLoader(train_abnormal_loader,
                                                       batch_size = self.train_flags.batch_size,
                                                       shuffle=True, num_workers = 8)
        self.test_normal_loader = torch.utils.data.DataLoader(test_normal_loader, 
                                                        batch_size = self.train_flags.batch_size,
                                                        shuffle=True, num_workers = 8)
        self.test_abnormal_loader = torch.utils.data.DataLoader(test_abnormal_loader,
                                                       batch_size = self.train_flags.batch_size,
                                                       shuffle=True, num_workers = 8)

    def train_model(self):
        print('[i]  Function : train_model')
        dir_train_summary = os.path.join(self.dir_result, 'train_summary')
        dir_test_summary = os.path.join(self.dir_result, 'test_summary')
        dir_model = os.path.join(self.dir_result, 'model')
        self._make_dir(dir_train_summary) 
        self._make_dir(dir_test_summary)
        self._make_dir(dir_model)
        max_epoch = self.train_flags.max_epoch
        train_writer = SummaryWriter(dir_train_summary)
        test_writer = SummaryWriter(dir_test_summary)
        for epoch in range(max_epoch):
            for iteration, data in enumerate(self.train_loader, 0):
                self.optimizer.zero_grad()
                img, label = data
                img = img.float().to(self.device)
                if epoch == 0 and iteration == 0:
                    train_writer.add_graph(self.net, img)
                label = label.to(self.device)
                outputs = self.net(img)
                _, predicted = torch.max(outputs.data, axis=1)
                train_loss = self.criterion(outputs, label)
                train_loss.backward()
                self.optimizer.step()
                train_avg_loss = train_loss / label.size(0)
                train_acc = (predicted == label).sum().item() / label.size(0)
                train_writer.add_scalar('batch_loss', train_loss.item(),
                                        epoch*len(self.train_loader) + iteration)
                train_writer.add_scalar('batch_acc', train_acc,
                                        epoch*len(self.train_loader) + iteration)
                if iteration == len(self.train_loader) - 1:
                    def loader_acc(loader):
                        total_num, correct_num, loss_all = 0, 0, 0
                        for iteration, data in enumerate(loader, 0):
                            with torch.no_grad():
                                img, label = data
                                img = img.float().to(self.device)
                                label = label.to(self.device)
                                outputs = self.net(img)
                                train_loss = self.criterion(outputs, label)
                                _, predicted = torch.max(outputs.data, axis=1)
                                loss_all += train_loss.item()
                                correct_num += (predicted == label).sum().item()
                                total_num += label.size(0)
                        total_acc = correct_num / total_num
                        loss_avg = loss_all / total_num
                        return total_acc, loss_avg
                    train_acc, train_loss = loader_acc(self.train_loader)
                    train_writer.add_scalar('total_acc', train_acc,
                                            (epoch + 1) * len(self.train_loader))
                    train_writer.add_scalar('total_loss', train_loss,
                                            (epoch + 1) * len(self.train_loader))
                    train_normal_acc, train_normal_loss = loader_acc(self.train_normal_loader)
                    train_writer.add_scalar('normal_acc', train_normal_acc,
                                            (epoch + 1) * len(self.train_loader))
                    train_writer.add_scalar('normal_loss', train_normal_loss,
                                            (epoch + 1) * len(self.train_loader))
                    train_abnormal_acc, train_abnormal_loss = loader_acc(self.train_abnormal_loader)
                    train_writer.add_scalar('abnormal_acc', train_abnormal_acc,
                                            (epoch + 1) * len(self.train_loader))
                    train_writer.add_scalar('abnormal_loss', train_abnormal_loss,
                                            (epoch + 1) * len(self.train_loader))
                    test_acc, test_loss = loader_acc(self.test_loader)
                    test_writer.add_scalar('total_acc', test_acc,
                                            (epoch + 1) * len(self.train_loader))
                    test_writer.add_scalar('total_loss', test_loss,
                                            (epoch + 1) * len(self.train_loader))
                    test_normal_acc, test_normal_loss = loader_acc(self.test_normal_loader)
                    test_writer.add_scalar('normal_acc', test_normal_acc,
                                            (epoch + 1) * len(self.train_loader))
                    test_writer.add_scalar('normal_loss', test_normal_loss,
                                            (epoch + 1) * len(self.train_loader))
                    test_abnormal_acc, test_abnormal_loss = loader_acc(self.test_abnormal_loader)
                    test_writer.add_scalar('abnormal_acc', test_abnormal_acc,
                                            (epoch + 1) * len(self.train_loader))
                    test_writer.add_scalar('abnormal_loss', test_abnormal_loss,
                                            (epoch + 1) * len(self.train_loader))



    def _make_dir(self, dir_):
        if not os.path.exists(dir_):
            os.mkdir(dir_)


if __name__ == '__main__':
    sys_flags = flags_module.get_sys_flags()
    train_flags = flags_module.get_train_flags()
    np.random.seed(sys_flags.random_seed)
    train_class()
