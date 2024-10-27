import matplotlib.pyplot as plt
import torch

class TrainingVisualizer:
    def __init__(self):
        self.train_losses = []
        self.valid_losses = []
        self.train_accs_task1 = []
        self.valid_accs_task1 = []
        self.train_accs_task2 = []
        self.valid_accs_task2 = []

    def update(self, train_loss, valid_loss, train_acc, valid_acc):
        """记录每个epoch的损失和准确率"""
        self.train_losses.append(train_loss.cpu().detach() if torch.is_tensor(train_loss) else train_loss)
        self.valid_losses.append(valid_loss.cpu().detach() if torch.is_tensor(valid_loss) else valid_loss)

        # 任务1和任务2的准确率分别记录
        train_acc_task1, train_acc_task2 = train_acc
        valid_acc_task1, valid_acc_task2 = valid_acc

        self.train_accs_task1.append(train_acc_task1.cpu().detach() if torch.is_tensor(train_acc_task1) else train_acc_task1)
        self.valid_accs_task1.append(valid_acc_task1.cpu().detach() if torch.is_tensor(valid_acc_task1) else valid_acc_task1)
        self.train_accs_task2.append(train_acc_task2.cpu().detach() if torch.is_tensor(train_acc_task2) else train_acc_task2)
        self.valid_accs_task2.append(valid_acc_task2.cpu().detach() if torch.is_tensor(valid_acc_task2) else valid_acc_task2)

    def plot(self):
        """绘制训练和验证集的损失和任务1、任务2的准确率曲线"""
        epochs = range(1, len(self.train_losses) + 1)

        # 确保所有张量都被转换为 CPU 数据
        train_losses = [tl.cpu().numpy() if torch.is_tensor(tl) else tl for tl in self.train_losses]
        valid_losses = [vl.cpu().numpy() if torch.is_tensor(vl) else vl for vl in self.valid_losses]
        train_accs_task1 = [ta.cpu().numpy() if torch.is_tensor(ta) else ta for ta in self.train_accs_task1]
        valid_accs_task1 = [va.cpu().numpy() if torch.is_tensor(va) else va for va in self.valid_accs_task1]
        train_accs_task2 = [ta.cpu().numpy() if torch.is_tensor(ta) else ta for ta in self.train_accs_task2]
        valid_accs_task2 = [va.cpu().numpy() if torch.is_tensor(va) else va for va in self.valid_accs_task2]

        plt.figure(figsize=(12, 10))

        # 绘制损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(epochs, train_losses, 'r-', label='Training loss')
        plt.plot(epochs, valid_losses, 'b-', label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # 绘制任务1的准确率曲线
        plt.subplot(2, 2, 2)
        plt.plot(epochs, train_accs_task1, 'r-', label='Training Accuracy Task 1')
        plt.plot(epochs, valid_accs_task1, 'b-', label='Validation Accuracy Task 1')
        plt.title('Training and Validation Accuracy (Task 1)')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        # 绘制任务2的准确率曲线
        plt.subplot(2, 2, 3)
        plt.plot(epochs, train_accs_task2, 'r-', label='Training Accuracy Task 2')
        plt.plot(epochs, valid_accs_task2, 'b-', label='Validation Accuracy Task 2')
        plt.title('Training and Validation Accuracy (Task 2)')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig('./output/result.png')
        plt.show()
