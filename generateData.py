import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
import math
import h5py
from sklearn.preprocessing import MinMaxScaler


def deleteRest_label():
    # 加载数据
    data = scio.loadmat('./data/integrated_emg_label.mat')
    emg_data = data['emg_data']
    emg_label = data['label_data']

    # 过滤掉标签为0的休息段
    emg_data_index1 = [i for i in range(len(emg_label)) if np.any(emg_label[i] != 0)]
    emg_label_filtered = emg_label[emg_data_index1, :]
    emg_data_filtered = emg_data[emg_data_index1, :]

    # 返回过滤后的emg数据和标签
    return emg_data_filtered, emg_label_filtered


def split_data():
    # 获取过滤后的数据和标签
    all_emg_data, all_label = deleteRest_label()
    emgData = []
    emgLabel = []
    imageLength = 500
    step = 250  # 设置步长
    classes_start = 30  # 第一列标签从 30 开始
    classes_end = 44  # 第一列标签到 44，表示 15 个类别

    # 遍历标签 (从 30 到 44, 共15个类别)
    for i in range(classes_start, classes_end + 1):
        index = []

        # 查找标签中符合条件的样本（考虑两列标签的情况）
        for j in range(all_label.shape[0]):
            if all_label[j, 0] == i or all_label[j, 1] == i:
                index.append(j)

        if len(index) == 0:
            print(f"class {i} has no samples.")
            continue

        iemg = all_emg_data[index, :]
        length = math.floor((iemg.shape[0] - imageLength) / step)
        print(f"class {i} number of samples: {iemg.shape[0]}, windows: {length}")

        for j in range(length):
            start_index = step * j
            end_index = start_index + imageLength
            if end_index <= iemg.shape[0]:
                subImage = iemg[start_index:end_index, :]
                emgData.append(subImage)
                # 保留多目标标签 [label1, label2]
                emgLabel.append(all_label[index[0], :].tolist())

    # 转换为 numpy 数组
    emgData = np.array(emgData)
    emgLabel = np.array(emgLabel)

    print(emgData.shape)
    print(emgLabel.shape)

    # 检查标签中是否有 0
    if np.any(emgLabel == 0):
        print("Warning: Some labels contain 0.")
    else:
        print("No labels contain 0.")

    # 保存数据到 .h5 文件
    with h5py.File('./data/image_500_250_multilabel.h5', 'w') as file:
        file.create_dataset('emgData', data=emgData)
        file.create_dataset('emgLabel', data=emgLabel)

# 调用函数
split_data()

