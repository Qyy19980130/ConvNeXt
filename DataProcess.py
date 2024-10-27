import random
import h5py
from config import *
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

def read_h5File(filePath):
    random.seed(3)
    file = h5py.File(filePath, 'r')
    imageData = file['emgData'][:]
    imageLabel = file['emgLabel'][:]
    file.close()
    # 随机打乱数据和标签
    N = imageData.shape[0]
    index = np.random.permutation(N)
    data = imageData[index, :, :]
    label = imageLabel[index]
    # 对数据升维
    data = np.expand_dims(data, axis=1)
    label = label
    # label = convert_to_one_hot(label, 49)

    # 按照7:2:1的比例划分数据集
    num_train = round(N * 0.7)
    num_val = round(N * 0.2)

    X_train = data[:num_train, :, :, :]
    Y_train = label[:num_train]
    X_val = data[num_train:num_train + num_val, :, :, :]
    Y_val = label[num_train:num_train + num_val]
    X_test = data[num_train + num_val:, :, :, :]
    Y_test = label[num_train + num_val:]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test


class MyDataSet(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label  # 假设label是形状为 (N, 2)，其中有两列，分别对应两个任务的标签
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        emg_data = self.data[idx]
        emg_label1 = self.label[idx, 0]  # 第一个任务的标签
        emg_label2 = self.label[idx, 1]  # 第二个任务的标签
        emg_data = emg_data.transpose(1, 2, 0)  # 转换为 w*h*c
        emg_label1 = torch.tensor(emg_label1, dtype=torch.long)
        emg_label2 = torch.tensor(emg_label2, dtype=torch.long)

        if self.transform:
            emg_data = self.transform(emg_data)

        return emg_data, emg_label1, emg_label2
        _label2  # 返回数据和两个标签


data_transforms = {
    'train':
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((150, 80))
        ]),
    'valid':
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((150, 80))
        ]),
    'test':
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((150, 80))
        ]),
}

if __name__ == '__main__':
    x_train, y_train, x_val, y_val, x_test, y_test = read_h5File(file_path)

    print('x_train: ', x_train.shape)
    print('y_train: ', y_train.shape)
    print('x_val: ', x_val.shape)
    print('y_val: ', y_val.shape)
    print('x_test: ', x_test.shape)
    print('y_test: ', y_test.shape)

    # 创建训练和验证数据集
    train_dataset = MyDataSet(x_train, y_train, transform=data_transforms['train'])
    val_dataset = MyDataSet(x_val, y_val, transform=data_transforms['valid'])
    test_dataset = MyDataSet(x_test, y_test, transform=data_transforms['test'])

    # 示例：获取第一个样本和两个标签
    sample, label1, label2 = train_dataset[0]
    print('Sample shape: ', sample.shape)
    print('Label1: ', label1)
    print('Label2: ', label2)

