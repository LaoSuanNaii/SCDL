import sys
sys.path.append('D:\\python\\pythoncode\\SSACNN\\smartcontract')
import os.path
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from model5 import SCDL
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 50
train_size = 0.85
label_dict = None
word_index = None
selection = 1360

def read_data():
    global label_dict, word_index
    datas = pd.read_csv("D:\\python\\pythoncode\\SSACNN\\smartcontract\\dataset\\train6.csv")
    values = datas["OPCODE"].values
    labels = datas["CATEGORY"].values
    if not os.path.exists("label_dict.json"):
        label_dict = {k: v for v, k in enumerate(set(list(labels)))}
        json_data = json.dumps(label_dict, indent=4)
        with open("label_dict.json", "w") as f:
            f.write(json_data)
    else:
        with open("label_dict.json", "r") as f:
            label_dict = json.load(f)

    labels = [label_dict[i] for i in labels]
    labels = np.int32(labels)
    indexes = []
    dict_label = {v: k for k, v in label_dict.items()}

    for i in range(len(label_dict)):
        index = list(np.where(labels == i)[0])
        if len(index) > selection:
            print(f"label：{dict_label[i]} -- {selection}")
            indexes.extend(random.sample(index, selection))
        else:
            print(f"label：{dict_label[i]} -- {len(index)}")
            indexes.extend(index)

    labels = labels[indexes]
    values = values[indexes]

    if not os.path.exists("word_index.json"):
        tokenizer = Tokenizer(oov_token="[CLS]")
        tokenizer.fit_on_texts(values)
        word_index = tokenizer.word_counts
        word_index = [(k, v) for k, v in word_index.items()]
        word_index.sort(key=lambda x: x[-1], reverse=True)
        word_index.insert(0, ("[PAD]", 0))
        word_index.insert(1, ("[CLS]", 0))
        word_index.insert(2, ("[SPE]", 0))
        word_index.insert(3, ("[UNK]", 0))
        word_index = {k[0]: v for v, k in enumerate(word_index)}
        data = json.dumps(word_index, indent=4, ensure_ascii=False)
        with open("word_index.json", "w", encoding="utf-8") as f:
            f.write(data)
    else:
        with open("word_index.json", "r", encoding="utf-8") as f:
            word_index = json.load(f)

    train_x, test_x, train_y, test_y = train_test_split(values, labels, train_size=train_size, stratify=labels)

    return train_x, train_y, test_x, test_y


class FeelData(Dataset):
    def __init__(self, texts, label):
        self.text, self.label = texts, label

    def __getitem__(self, item):
        text = self.text[item]
        label = self.label[item]
        text = text.split(" ")
        value = [word_index["[CLS]"]]

        for i in text:
            try:
                value.append(word_index[i])
            except:
                continue
        return value, label

    @staticmethod
    def call_fc(batch):
        value = []
        label = []
        for x, y in batch:
            value.append(x)
            label.append(y)
        value = pad_sequences(value, padding="post")
        label = np.int64(label)
        if value.shape[-1] > 512:
            value = value[:, :512]
        return torch.IntTensor(value), torch.LongTensor(label)

    def __len__(self):
        return len(self.text)

# [5, 3, 4, 10, 16, 12, 3, 4, 3]
def train_model(model_init):
    a = int(model_init[0])
    batch_size = 2**a
    b = int(model_init[1])
    c = int(model_init[2])
    lr = b*(10**(-c))
    train_x, train_y, test_x, test_y = read_data()
    train_data = FeelData(train_x, train_y)
    train_data = DataLoader(train_data, shuffle=True, batch_size=batch_size, pin_memory=True,
                            collate_fn=train_data.call_fc)
    test_data = FeelData(test_x, test_y)
    test_data = DataLoader(test_data, shuffle=False, batch_size=batch_size, pin_memory=True,
                           collate_fn=test_data.call_fc)

    d_model = 512
    num_layers = int(model_init[3])  # transformer encoder的层数 10layers
    n_head = int(model_init[4])
    block_layers = int(model_init[5])
    kernel_1 = int(model_init[6])
    kernel_2 = int(model_init[7])
    kernel_3 = int(model_init[8])

    model = SCDL(d_model=d_model, num_layers=num_layers, n_head=n_head, block_layers=block_layers,
                 out_dim=len(label_dict), v_size=len(word_index), input_dim=d_model,
                 kernel_1=kernel_1, kernel_2=kernel_2, kernel_3=kernel_3)
    if torch.cuda.device_count() > 0:
        print('let us use', torch.cuda.device_count(), 'GPUS!!!')
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fc = nn.CrossEntropyLoss()
    loss_old = 100
    acc_max1 = 0
    acc_max2 = 0
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_data)
        loss_all1 = 0
        acc_all1 = 0
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_batch1 = loss_fc(out, y)
            loss_batch1.backward()
            optimizer.step()  # 梯度下降
            optimizer.zero_grad()  # 清空梯度
            loss_all1 += loss_batch1.item()  # 累加loss
            loss_epoch1 = loss_all1 / (step + 1)  # 求当前平均loss
            acc_batch1 = torch.mean((y == torch.argmax(out, dim=-1)).float())  # 求出准确率
            acc_all1 += acc_batch1  # 累加acc
            acc_epoch1 = acc_all1 / (step + 1)  # 求当前平均acc
            # summary(model,x)
            s = (
                "train => epoch:{} - step:{} - loss_batch:{:.4f} - loss_epoch:{:.4f} - acc_batch:{:.4f} - acc_epoch:{:.4f}".format(
                    epoch, step, loss_batch1,
                    loss_epoch1, acc_batch1, acc_epoch1))  # 构建进度条信息
            pbar.set_description(s)  # 显示进度条

        # 测试
        pre_label = torch.tensor([], dtype=int).to(device)
        with torch.no_grad():  # 测试模式
            # print("pre_label len")
            # print(len(pre_label))
            pbar = tqdm(test_data)  # 将数据放入进度条中
            loss_all2 = 0  # 初始化loss_all
            acc_all2 = 0  # 初始化acc_all
            for step, (x, y) in enumerate(pbar):  # 一一遍历取出值
                x, y = x.to(device), y.to(device)  # 放入device环境下
                out = model(x)  # 放入网络，预测
                loss_batch2 = loss_fc(out, y)  # 计算损失
                loss_all2 += loss_batch2.item()  # 累加loss
                loss_epoch2 = loss_all2 / (step + 1)  # 求当前平均loss
                acc_batch2 = torch.mean((y == torch.argmax(out, dim=-1)).float())  # 求出准确率
                pre = torch.argmax(out, dim=-1)
                pre_label = torch.cat([pre_label, pre], 0)
                acc_all2 += acc_batch2  # 累加acc
                acc_epoch2 = acc_all2 / (step + 1)  # 求当前平均acc
                s = (
                    "test => epoch:{} - step:{} - loss_batch:{:.4f} - loss_epoch:{:.4f} - acc_batch:{:.4f} - acc_epoch:{:.4f}".format(
                        epoch, step, loss_batch2,
                        loss_epoch2, acc_batch2, acc_epoch2))  # 构建进度条信息
                pbar.set_description(s)  # 显示进度条

        # 求训练过程的最大准确率
        if acc_max1 < acc_epoch1:
            acc_max1 = acc_epoch1
        if acc_max2 < acc_epoch2:
            acc_max2 = acc_epoch2
        print("\nacc_max1:{:.4f}".format(acc_max1))
        print("acc_max2:{:.4f}".format(acc_max2))

        # 储存准确率和loss数据
        loss_epoch1t = torch.tensor(loss_epoch1)
        loss_epoch2t = torch.tensor(loss_epoch2)
        train_acc.append(acc_epoch1.item())  # 准确率加入到列表中
        train_loss.append(loss_epoch1t.item())  # 损失加入到列表中
        test_loss.append(loss_epoch2t.item())  # 损失加入到列表中
        test_acc.append(acc_epoch2.item())  # 准确率加入到列表中
        with open("./train_loss.txt", 'w') as train_los:
            train_los.write(str(train_loss))
        with open("./train_acc.txt", 'w') as train_ac:
            train_ac.write(str(train_acc))
        with open("./test_loss.txt", 'w') as test_los:
            test_los.write(str(test_loss))
        with open("./test_acc.txt", 'w') as test_ac:
            test_ac.write(str(test_acc))

        # 权重文件
        if loss_old > loss_epoch2:  # 如果loss_time小于loss_old
            loss_old = loss_epoch2  # 赋值
            # torch.save(model.state_dict(), "model.pkl")  # 保存权重文件
            torch.save(model.state_dict(), "test.pkl")  # 保存权重文件

    # 混淆矩阵
    pre_label = torch.tensor(pre_label, device='cpu')
    cm = confusion_matrix(y_true=test_y, y_pred=pre_label)
    print("Cofusion Matrix: ")
    print(cm)
    # 模型评估报告
    cr = classification_report(y_true=test_y, y_pred=pre_label)
    q = classification_report(y_true=test_y, y_pred=pre_label, output_dict=True)
    print("classification_report: ")
    print(cr)
    print(q)
    # print(test_y)
    # print(pre_label)
    # 可视化混淆矩阵
    cmp = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmp.plot(cmap=plt.cm.Blues)
    plt.show()

    # params = list(model.parameters())
    # for param in params:
    #     print(param.shape)

    return loss_epoch2  # ssa优化的参数


# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")  # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asfarray(data, float)


if __name__ == '__main__':
    print('Start train')
    train_model([4, 3, 5, 8, 8, 6, 3, 4, 3])

    # 画loss图
    train_loss_path = r"D:\python\pythoncode\SSACNN\smartcontract\train_loss.txt"  # 存储文件路径
    test_loss_path = r"D:\python\pythoncode\SSACNN\smartcontract\test_loss.txt"
    y_train_loss = data_read(train_loss_path)  # loss值，即y轴
    y_test_loss = data_read(test_loss_path)
    x_train_loss = range(len(y_train_loss))  # loss的数量，即x轴
    x_test_loss = range(len(y_test_loss))  # loss的数量，即x轴
    plt.figure()
    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('epoch')  # x轴标签
    plt.ylabel('loss')  # y轴标签
    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.plot(x_test_loss, y_test_loss, linewidth=1, linestyle="solid", label="test loss")
    plt.legend()
    plt.title('Loss curve')
    plt.show()

    # 画acc图
    train_acc_path = r"D:\python\pythoncode\SSACNN\smartcontract\train_acc.txt"
    test_acc_path = r"D:\python\pythoncode\SSACNN\smartcontract\test_acc.txt"
    y_train_acc = data_read(train_acc_path)
    y_test_acc = data_read(test_acc_path)
    x_train_acc = range(len(y_train_acc))  # 训练阶段准确率的数量，即x轴
    x_test_acc = range(len(y_test_acc))  # 训练阶段准确率的数量，即x轴
    plt.figure()
    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.xlabel('epochs')  # x轴标签
    plt.ylabel('accuracy')  # y轴标签
    # 以x_train_acc为横坐标，y_train_acc为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 增加参数color='red',这是红色。
    plt.plot(x_train_acc, y_train_acc, linewidth=1, linestyle="solid", label="train acc")
    plt.plot(x_test_acc, y_test_acc, linewidth=1, linestyle="solid", label="test acc")
    plt.legend()
    plt.title('Accuracy curve')
    plt.show()
