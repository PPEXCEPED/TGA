# -*- coding = utf-8 -*-
# @Time : 2023/1/11 21:37
# @Author : 头发没了还会再长
# @File : FinalClassfier.py
# @Software : PyCharm
import torch
import numpy as np
import process.DataSet as LoadData
import process.TextFeatureEmbedding as TextFeature
from process import ImageFeature
import process.FuseAllFeature as StageFusion


class ClassificationLayer(torch.nn.Module):
    def __init__(self, dropout_rate=0, dim=0):
        super(ClassificationLayer, self).__init__()
        self.Linear_1 = torch.nn.Linear(dim, 256)
        self.linear_2 = torch.nn.Linear(256, 128)
        self.Linear_3 = torch.nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, input):
        hidden = self.Linear_1(input)
        hidden = self.dropout(hidden)
        hidden = torch.relu(self.linear_2(hidden))
        output = torch.sigmoid(self.Linear_3(hidden))
        return output


class ClassificationLayer_1(torch.nn.Module):
    def __init__(self, dropout_rate=0, dim=0):
        super(ClassificationLayer_1, self).__init__()
        self.Linear_1 = torch.nn.Linear(dim, 128)
        # self.linear_2 = torch.nn.Linear(256,128)
        self.Linear_2 = torch.nn.Linear(128, 1)
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, input):
        hidden = self.Linear_1(input)
        hidden = self.dropout(hidden)
        # hidden = torch.relu(self.linear_2(hidden))
        output = torch.sigmoid(self.Linear_2(hidden))
        return output


class finallClass(torch.nn.Module):
    def __init__(self, dim=0):
        super(finallClass, self).__init__()
        self.linear_1 = torch.nn.Linear(dim, 6)
        self.norm_1 = torch.nn.BatchNorm2d(6)
        self.linear_2 = torch.nn.Linear(6, 4)
        self.norm_2 = torch.nn.BatchNorm2d(4)
        self.linear_3 = torch.nn.Linear(4, 2)
        self.norm_3 = torch.nn.BatchNorm2d(2)
        self.linear_4 = torch.nn.Linear(2, 1)

    def forward(self, input):
        hidden = self.linear_1(input)
        hidden = self.norm_1(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.norm_2(hidden)
        hidden = self.linear_3(hidden)
        hidden = self.norm_3(hidden)
        output = torch.sigmoid(self.linear_4(hidden))
        return output
