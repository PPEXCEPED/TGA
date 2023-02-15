# -*- coding = utf-8 -*-
# @Time : 2023/1/5 14:36
# @Author : 头发没了还会再长
# @File : ImageFeature.py
# @Software : PyCharm
import process.DataSet as DataSet
import torch
class ExtractImageFeature(torch.nn.Module):
    def __init__(self):
        super(ExtractImageFeature, self).__init__()
        # 1024->512
        self.Linear = torch.nn.Linear(1024, 1024)

    def forward(self, input):
        input=input.permute(1,0,2)#[196, 64, 1024]
        # print('input text:',input.shape)
        output=list()
        for i in range(196):
            sub_output=torch.nn.functional.relu(self.Linear(input[i])) #[64, 1024]
            output.append(sub_output)
        output=torch.stack(output)#[196, 64, 1024]
        mean=torch.mean(output,0)
        return mean, output

if __name__ == "__main__":
    test=ExtractImageFeature()
    for text_index,image_feature,group,id in DataSet.train_loader:
        mean, output=test(image_feature)
        print(mean.shape)  # [64, 1024]
        print(output.shape)  # [196, 64, 1024]
        break
