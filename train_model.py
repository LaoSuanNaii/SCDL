import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from model import DL4SC
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
from sklearn.model_selection import train_test_split


device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 50
train_size = 0.85
label_dict = None
word_index = None
selection = 1360

def read_data():
    global label_dict, word_index
    datas = pd.read_csv("data.csv")
    values = datas["OPCODE"].values
    labels = datas["LABEL"].values
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
        word_index = {k[0]: v for v, k in enumerate(word_index)}
        data = json.dumps(word_index, indent=4, ensure_ascii=False)
        with open("word_index.json", "w", encoding="utf-8") as f:
            f.write(data)
    else:
        with open("word_index.json", "r", encoding="utf-8") as f:
            word_index = json.load(f)

    train_x, test_x, train_y, test_y = train_test_split(values, labels, train_size=train_size)

    return train_x, train_y, test_x, test_y


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
    num_layers = int(model_init[3])
    n_head = int(model_init[4])
    block_layers = int(model_init[5])
    kernel_1 = int(model_init[6])
    kernel_2 = int(model_init[7])
    kernel_3 = int(model_init[8])

    model = DL4SC(d_model=d_model, num_layers=num_layers, n_head=n_head, block_layers=block_layers,
                 out_dim=len(label_dict), v_size=len(word_index), input_dim=d_model,
                 kernel_1=kernel_1, kernel_2=kernel_2, kernel_3=kernel_3)
    if torch.cuda.device_count() > 0:
        print('let us use', torch.cuda.device_count(), 'GPUS!!!')
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fc = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_data)
        loss_all1 = 0
        acc_all1 = 0
        for step, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_batch1 = loss_fc(out, y)
            loss_batch1.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_all1 += loss_batch1.item()
            loss_epoch1 = loss_all1 / (step + 1)
            acc_batch1 = torch.mean((y == torch.argmax(out, dim=-1)).float())
            acc_all1 += acc_batch1
            acc_epoch1 = acc_all1 / (step + 1)
            s = (
                "train => epoch:{} - step:{} - loss_batch:{:.4f} - loss_epoch:{:.4f} - acc_batch:{:.4f} - acc_epoch:{:.4f}".format(
                    epoch, step, loss_batch1,
                    loss_epoch1, acc_batch1, acc_epoch1))
            pbar.set_description(s)

        with torch.no_grad():
            pbar = tqdm(test_data)
            loss_all2 = 0
            acc_all2 = 0
            for step, (x, y) in enumerate(pbar):
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss_batch2 = loss_fc(out, y)
                loss_all2 += loss_batch2.item()
                loss_epoch2 = loss_all2 / (step + 1)
                acc_batch2 = torch.mean((y == torch.argmax(out, dim=-1)).float())
                pre = torch.argmax(out, dim=-1)
                pre_label = torch.cat([pre_label, pre], 0)
                acc_all2 += acc_batch2
                acc_epoch2 = acc_all2 / (step + 1)
                s = (
                    "test => epoch:{} - step:{} - loss_batch:{:.4f} - loss_epoch:{:.4f} - acc_batch:{:.4f} - acc_epoch:{:.4f}".format(
                        epoch, step, loss_batch2,
                        loss_epoch2, acc_batch2, acc_epoch2))
                pbar.set_description(s)
    return loss_epoch2



if __name__ == '__main__':
    print('Start train')
    train_model([5, 3, 4, 10, 16, 12, 3, 4, 3])
