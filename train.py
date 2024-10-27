import torch
import torch.nn as nn
from torchsummary import summary
import torch.optim as optim
from DataProcess import *
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
import copy
from LossHistory import TrainingVisualizer
from models.convnext import *


def adjust_labels(y_train, y_val, y_test):
    # 调整第一个标签，将其范围从 [30, 44] 调整为 [0, 14]
    y_train[:, 0] = y_train[:, 0] - 30
    y_val[:, 0] = y_val[:, 0] - 30
    y_test[:, 0] = y_test[:, 0] - 30

    # 调整第二个标签，将其范围从 [1, 2, 3] 调整为 [0, 1, 2]
    y_train[:, 1] = y_train[:, 1] - 1
    y_val[:, 1] = y_val[:, 1] - 1
    y_test[:, 1] = y_test[:, 1] - 1

    return y_train, y_val, y_test

x_train, y_train, x_val, y_val, x_test, y_test = read_h5File(file_path)
print('x_train: ', x_train.shape)
print('y_train: ', y_train.shape)
print('x_val: ', x_val.shape)
print('y_val: ', y_val.shape)
print('x_test: ', x_test.shape)
print('y_test: ', y_test.shape)

y_train, y_val, y_test = adjust_labels(y_train, y_val, y_test)


train_dataset = MyDataSet(x_train, y_train, transform=data_transforms['train'])
val_dataset = MyDataSet(x_val, y_val, transform=data_transforms['valid'])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
dataloaders = {'train': train_loader, 'valid': val_loader}
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU...')
else:
    print('CUDA is available. Training on GPU...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, dataloaders, criterion1, criterion2, optimizer, num_epochs, filename='./output/model.pth'):
    since = time.time()
    best_acc = 0
    model.to(device)
    val_acc_history = []
    train_acc_history = []
    train_losses = []
    valid_losses = []
    LRs = [optimizer.param_groups[0]['lr']]
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects_task1 = 0
            running_corrects_task2 = 0

            with tqdm(total=len(dataloaders[phase]), desc=f'{phase} Epoch {epoch + 1}/{num_epochs}',
                      unit='batch') as pbar:
                for inputs, labels1, labels2 in dataloaders[phase]:  # 两个任务的标签
                    inputs = inputs.permute(0, 1, 3, 2).to(device)
                    labels1 = labels1.to(device)  # 第一个任务的标签
                    labels2 = labels2.to(device)  # 第二个任务的标签

                    optimizer.zero_grad()  # 清零梯度

                    with torch.set_grad_enabled(phase == 'train'):  # 只有训练时才计算梯度
                        outputs1, outputs2 = model(inputs)  # 模型的两个输出，分别对应任务1和任务2

                        # 分别计算任务1和任务2的损失
                        loss1 = criterion1(outputs1, labels1)
                        loss2 = criterion2(outputs2, labels2)

                        # 组合两个任务的损失，可以根据实际情况设置权重
                        loss = loss1 + loss2

                        _, preds1 = torch.max(outputs1, 1)  # 预测任务1
                        _, preds2 = torch.max(outputs2, 1)  # 预测任务2

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 计算损失
                    running_loss += loss.item() * inputs.size(0)  # 损失乘以batch大小

                    # 分别计算每个任务的正确预测数
                    running_corrects_task1 += torch.sum(preds1 == labels1.data)
                    running_corrects_task2 += torch.sum(preds2 == labels2.data)

                    pbar.update(1)

            # 计算每个任务的平均损失和准确率
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc_task1 = running_corrects_task1.double() / len(dataloaders[phase].dataset)
            epoch_acc_task2 = running_corrects_task2.double() / len(dataloaders[phase].dataset)

            time_elapsed = time.time() - since
            print('Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print('{} Loss: {:.4f} Task1 Acc: {:.4f} Task2 Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc_task1,
                                                                               epoch_acc_task2))

            # 保存最佳模型
            if phase == 'valid' and (epoch_acc_task1 + epoch_acc_task2) / 2 > best_acc:
                best_acc = (epoch_acc_task1 + epoch_acc_task2) / 2
                best_model_wts = copy.deepcopy(model.state_dict())
                state = {
                    'state_dict': model.state_dict(),
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(best_model_wts, filename)
                print(f'Best model saved with average accuracy: {best_acc:.4f}')

            if phase == 'valid':
                val_acc_history.append((epoch_acc_task1, epoch_acc_task2))
                valid_losses.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append((epoch_acc_task1, epoch_acc_task2))
                train_losses.append(epoch_loss)

        print('Optimizer learning rate:{:.7f}'.format(optimizer.param_groups[0]['lr']))
        LRs.append(optimizer.param_groups[0]['lr'])

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs


if __name__ == '__main__':
    my_model = ConvNeXt(
        in_chans=1,
        num_classes1=15,
        num_classes2=3,
        depths=[3, 3, 9, 3],  # 每个 stage 包含的 block 数量
        dims=[96, 192, 384, 768],  # 每个 stage 的通道数
        drop_path_rate=0.1,  # DropPath 概率
        layer_scale_init_value=1e-6,  # 层缩放的初始值
        head_init_scale=1.0  # 线性分类头的初始化缩放
    ).to(device)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(my_model.parameters(), lr=lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    # summary(my_model, (12, 1, 500))

    # 创建 LossHistory 实例
    visualizer = TrainingVisualizer()
    # 开始训练，传入两个损失函数
    best_model, val_acc_history, train_acc_history, valid_losses, train_losses, LRs = train(
        my_model, dataloaders,
        criterion1, criterion2,
        optimizer_ft,
        epochs
    )
    # 更新并绘制损失和准确率曲线
    for i in range(epochs):
        visualizer.update(train_losses[i], valid_losses[i], train_acc_history[i], val_acc_history[i])
    visualizer.plot()

