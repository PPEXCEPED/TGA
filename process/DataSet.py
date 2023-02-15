import os

import PIL.Image
import numpy as np
import torch
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tools import config

# 加载数据到内存
# 处理数据
# 定义Dataset 划分训练和验证集
# 定义DataLoader
root_path = '../data/'
TEXT_LENGTH = 75
TEXT_HIDDEN = 256


# load train/test data from disk to memory
def load_data(path):
    data_set = dict()
    file = open(os.path.join(root_path, "text_data", path + ".txt"), "rb")
    for line in file:
        content = eval(line)
        image = content[2]
        sentence = content[1]
        # domain_label = content[3]
        class_label = content[4]
        if os.path.isfile(os.path.join(root_path, "image_data", image + ".jpg")):
            if image in data_set:
                print(image)
            data_set[str(image)] = {"text": sentence, "class_label": class_label,
                                    "image_path": os.path.join(root_path, "image_data/", str(image) + '.jpg')}

    return data_set


root_path2 = "../data/weibo2"


def load_data2(mode):
    data_set = dict()
    tt_t_file = os.path.join(root_path2, 'test_text_with_label.npz')
    tt_i_file = os.path.join(root_path2, 'test_image_with_label.npz')
    tr_t_file = os.path.join(root_path2, 'train_text_with_label.npz')
    tr_i_file = os.path.join(root_path2, 'train_image_with_label.npz')
    tt1 = np.load(tt_t_file)
    tt1_label = tt1['label']
    tt1 = tt1['data']
    tt2 = np.load(tt_i_file)
    tt2 = tt2['data']
    tr1 = np.load(tr_t_file)
    tr1_label = tr1['label']
    tr1 = tr1['data']
    tr2 = np.load(tr_i_file)
    tr2 = tr2['data']
    # # print(len(tr1))
    # print(tt1[0].shape, tt1[0].type)
    # text = tt1[0].view(-1)
    # print(text)
    if mode == 'test':
        for i in range(len(tt1)):

            data_set[i] = {'text': tt1[i].flatten()[:TEXT_LENGTH], 'class_label': tt1_label[i], 'image': tt2[i]}
    else:
        for i in range(len(tr1)):
            data_set[i] = {'text': tr1[i], 'class_label': tr1_label[i], 'image': tr2[i]}

    return data_set


train_data_set = load_data("minitrain")
# test_data_set = load_data2("test")
test_data_set = load_data("test")


# load text vectors {text: idx}
def load_word_index():
    with open(os.path.join(root_path, "text_embedding/vocabs.txt"), "rb") as f:
        for line in f:
            word2index = eval(line)
    return word2index


word2index = load_word_index()


# dataset and dataloader
class RumorDataset():
    def __init__(self, data):
        self.data = data
        self.image_ids = list(data.keys())
        for id in data.keys():
            self.data[id]["image_path"] = os.path.join("../data/image_data/", str(id) + ".jpg")

        for id in data.keys():
            text = self.data[id]['text'].split()
            text_index = torch.empty(TEXT_LENGTH, dtype=torch.long)
            cur_len = len(text)
            for i in range(TEXT_LENGTH):
                if i >= cur_len:
                    text_index[i] = word2index["<pad>"]
                elif text[i] in word2index:
                    text_index[i] = word2index[text[i]]
                else:
                    text_index[i] = word2index['<pad>']
            self.data[id]['text_index'] = text_index

    def __image_feature_loader(self, id):
        image_feature = np.load(os.path.join("../data/image_feature_data", str(id) + ".npy"))
        return torch.from_numpy(image_feature)

    def __text_index_loader(self, id):
        return self.data[id]["text_index"]

    def text_loader(self, id):
        return self.data[id]['text']

    def image_loader(self, id):
        path = self.data[id]['image_path']
        img_pil = PIL.Image.open(path)
        transform = transforms.Compose([transforms.Resize((448, 448)),
                                        transforms.ToTensor()
                                        ])
        return transform(img_pil)

    def __getitem__(self, index):
        id = self.image_ids[index]
        text_index = self.__text_index_loader(id)
        image_feature = self.__image_feature_loader(id)
        class_label = self.data[id]['class_label']
        return text_index, image_feature, class_label, id

    def __len__(self):
        return len(self.image_ids)


class RumorDataset2():
    def __init__(self, data):
        self.data = data

    def __image_feature_loader(self, id):
        image_feature = np.load(os.path.join("../data/image_feature_data2", "tensor(" + str(id) + ")" + ".npy"))
        return torch.from_numpy(image_feature)

    def __getitem__(self, index):
        text = self.data[index]['text']
        # text1 = text.view(-1)
        image = self.__image_feature_loader(index)
        label = self.data[index]['class_label']
        return text, image, label, index

    def __len__(self):
        return len(self.data)


def train_val_split(train_data, val_ratio):
    train_val_count = [int(len(train_data) * val_ratio), 0]
    train_val_count[1] = int(len(train_data)) - train_val_count[0]
    val_set, train_set = random_split(train_data, train_val_count, generator=torch.Generator().manual_seed(43))
    return train_set, val_set


train_data_set = RumorDataset(train_data_set)
# test_data_set = RumorDataset2(test_data_set)
test_data_set = RumorDataset(test_data_set)
train_data_set, val_data_set = train_val_split(train_data_set, config.config['val_ratio'])


# DataLoader
def dataloader(data_set, batch_size, mode='train'):
    dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=(mode == 'train'))
    return dataloader


train_loader = dataloader(train_data_set, config.config['batch_size'], 'train')
test_loader = dataloader(test_data_set, config.config['batch_size'], 'test')
val_loader = dataloader(val_data_set, config.config['batch_size'], 'val')

for text_idx, image_feature, class_label, id in train_loader:
    print('text:', text_idx.shape, text_idx.type())  # torch.Size([32, 75]) torch.LongTensor 每一个句子用75个数表示 一个batch有64个句子
    print("image feature：", image_feature.shape, image_feature.type())  # torch.Size([32, 196, 1024]) torch.FloatTensor
    print("label:", class_label.shape,
          class_label.type)  # torch.Size([32]) <built-in method type of Tensor object at 0x0000028E6BC9D720>
    break
