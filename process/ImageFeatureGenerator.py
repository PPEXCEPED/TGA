import math
import os
import time

import PIL.Image
import torch
from PIL import Image
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import numpy as np
from PIL import ImageFile
import models.vit_model as vit
# from process.DataSet import *
from models.vit_model import vit_large_patch32_224_in21k as create_model
from tools.utils import read_split_data, train_one_epoch, evaluate
from tools.config import config

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_path = "../data"
TEXT_LENGTH=75
TEXT_HIDDEN=256


def load_data():
    data_set = dict()
    for dataset in ['minitrain']:
        file = open(os.path.join("../data/text_data", dataset + ".txt"), "rb")
        for line in file:
            content = eval(line)
            image = content[2]
            sentence = content[1]
            group = content[4]
            if os.path.isfile(os.path.join(root_path, "image_data/", image + ".jpg")):
                data_set[str(image)] = {"text": sentence, "group": group}
    return data_set


data_set = load_data()

image_feature_floder = "image_feature_data"

class pretrain_data_set(Dataset):
    def __init__(self, data):
        # super(pretrain_data_set, self).__init__()
        self.data = data
        self.image_ids = list(data.keys())
        for id in data.keys():
            self.data[id]["image_path"] = os.path.join(root_path, "image_data/", str(id)+".jpg")

    def __iamge_loader(self, id):
        path = self.data[id]["image_path"]
        img_pil = PIL.Image.open(path)
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img_pil)
        return img_tensor

    def __getitem__(self, idx):
        id = self.image_ids[idx]
        image = self.__iamge_loader(id)
        return id, image

    def __len__(self):
        return len(self.image_ids)

sub_image_size=32 #448/14
sub_graph_preprocess = transforms.Compose([
    transforms.ToPILImage(mode=None),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
all_pretrain_dataset=pretrain_data_set(data_set)
all_pretrain_loader = DataLoader(all_pretrain_dataset,batch_size=64)

# 生成图像特征
def VIT_predictor():
    model = create_model(num_classes=config['num_classes'], has_logits=False).to(device)
    for param in model.parameters():
        param.requires_grad = False

    # 已下载好的预训练权重
    weights_path = "../data/weights/vit_large_patch32_224_in21k.pth"
    if os.path.exists(weights_path):
        assert os.path.exists(weights_path), "weights file: '{}' not exist.".format(weights_path)
        weights_dict = torch.load(weights_path, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    model.eval()

    vit_output_path="../data/image_feature_data"
    if os.path.exists(vit_output_path) is False:
        os.makedirs(vit_output_path)
    with torch.no_grad():
        total = len(all_pretrain_loader) * all_pretrain_loader.batch_size
        count = 0
        time_s = time.perf_counter()
        for img_index, img in all_pretrain_loader:
            sub_img_output = list()
            for column in range(14):
                for row in range(14):
                    # resize image from (32,32) to (256,256)
                    sub_image_original = img[:, :, sub_image_size * row:sub_image_size * (row + 1),
                                         sub_image_size * column:sub_image_size * (column + 1)]
                    sub_image_normalized = torch.stack(
                        list(map(lambda image: sub_graph_preprocess(image), sub_image_original)), dim=0)
                    output = model(sub_image_normalized.to(device))
                    sub_img_output.append(output.to("cpu").numpy())
            sub_img_output = np.array(sub_img_output).transpose([1, 0, 2])
            # save averaged attribute to "resnet50_output", same name as the image
            for index, sub_img_index in enumerate(img_index):
                np.save(os.path.join(vit_output_path, str(sub_img_index)), sub_img_output[index])
            time_e = time.perf_counter()
            count += all_pretrain_loader.batch_size
            total_time = time_e - time_s
            print(
                f"Completed {count}/{total} time left={int((total - count) * total_time / count / 60 / 60)}:{int((total - count) * total_time / count / 60 % 60)}:{int((total - count) * total_time / count % 60)} speed={round(total_time / count, 3)}sec/image")

VIT_predictor()

# from torchvision.models import ResNet50_Weights
#
# ImageFile.LOAD_TRUNCATED_IMAGES = True
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# WORKING_PATH="./"
# TEXT_LENGTH=75
# TEXT_HIDDEN=256
# image_feature_folder="image_feature_data"
# all_pretrain_dataset=pretrain_data_set(data_set)
# """
# generate data
# """
# class Identity(torch.nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()
#     def forward(self, x):
#         return x
# def resnet50_predictor():
#     # extract the input for last fc layer in resenet50
# 	resnet50=torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
# 	for param in resnet50.parameters():
# 		param.requires_grad = False
# 	resnet50.fc = Identity()
# 	resnet50 = resnet50.to(device)
# 	resnet50.eval()
#     # save the output in .npy file
# 	resnet50_output_path=os.path.join(WORKING_PATH,image_feature_folder)
# 	if not os.path.exists(resnet50_output_path):
# 		os.makedirs(resnet50_output_path)
# 	with torch.no_grad():
# 		total=len(all_pretrain_loader)*all_pretrain_loader.batch_size
# 		count=0
# 		time_s=time.perf_counter()
# 		for img_index,img in all_pretrain_loader:
#
#             # seperate img(448,448) into 14*14 images with size (32,32)
#             # [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
#             # [14,15,16,17,18,................]
#             # [28,...]
#             # ...
#             # [182,....,195]
# 			sub_img_output=list()
# 			for column in range(14):
# 				for row in range(14):
#                     # resize image from (32,32) to (256,256)
# 					sub_image_original=img[:,:,sub_image_size*row:sub_image_size*(row+1),sub_image_size*column:sub_image_size*(column+1)]
# 					sub_image_normalized=torch.stack(list(map(lambda image:sub_graph_preprocess(image),sub_image_original)),dim=0)
# 					output=resnet50(sub_image_normalized.to(device))
# 					sub_img_output.append(output.to("cpu").numpy())
# 			sub_img_output=np.array(sub_img_output).transpose([1,0,2])
#             # save averaged attribute to "resnet50_output", same name as the image
# 			for index,sub_img_index in enumerate(img_index):
# 				np.save(os.path.join(resnet50_output_path,str(sub_img_index)),sub_img_output[index])
# 			time_e=time.perf_counter()
# 			count+=all_pretrain_loader.batch_size
# 			total_time=time_e-time_s
# 			print(f"Completed {count}/{total} time left={int((total-count)*total_time/count/60/60)}:{int((total-count)*total_time/count/60%60)}:{int((total-count)*total_time/count%60)} speed={round(total_time/count,3)}sec/image")
#
# # 32 is the minimum batch size can achieve best performance
# all_pretrain_loader = DataLoader(all_pretrain_dataset,batch_size=64)
#
# # it will take really long time to run...
# resnet50_predictor()
# if __name__=='main':
#     if os.path.exists("../data/weights") is False:
#         os.makedirs("../data/weights")
#
#     tb_writer = SummaryWriter()
#
#     data_transform = {
#         "train": transforms.Compose([transforms.RandomResizedCrop(224),
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
#         "val": transforms.Compose([transforms.Resize(256),
#                                    transforms.CenterCrop(224),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])}
#     train_dataset = train_data_set
#     val_dataset = val_data_set
#     batch_size = config['barch_size']
#     nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
#     print('Using {} dataloader workers every process'.format(nw))
#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                                batch_size=batch_size,
#                                                shuffle=True,
#                                                pin_memory=True,
#                                                num_workers=nw,
#                                                collate_fn=train_dataset.collate_fn)
#
#     val_loader = torch.utils.data.DataLoader(val_dataset,
#                                              batch_size=batch_size,
#                                              shuffle=False,
#                                              pin_memory=True,
#                                              num_workers=nw,
#                                              collate_fn=val_dataset.collate_fn)
#
#     model = create_model(num_classes=config['num_classes'], has_logits=False).to(device)
#
#     # 已下载好的预训练权重
#     weights_path = "../data/weights/vit_large_patch32_224_in21k.pth"
#     if os.path.exists(weights_path):
#         assert os.path.exists(weights_path), "weights file: '{}' not exist.".format(weights_path)
#         weights_dict = torch.load(weights_path, map_location=device)
#         # 删除不需要的权重
#         del_keys = ['head.weight', 'head.bias'] if model.has_logits \
#             else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
#         for k in del_keys:
#             del weights_dict[k]
#         print(model.load_state_dict(weights_dict, strict=False))
#
#     # if args.freeze_layers:
#     #     for name, para in model.named_parameters():
#     #         # 除head, pre_logits外，其他权重全部冻结
#     #         if "head" not in name and "pre_logits" not in name:
#     #             para.requires_grad_(False)
#     #         else:
#     #             print("training {}".format(name))
#
#     pg = [p for p in model.parameters() if p.requires_grad]
#     optimizer = optim.SGD(pg, lr=config['lr'], momentum=0.9, weight_decay=5E-5)
#     # Scheduler https://arxiv.org/pdf/1812.01187.pdf
#     lf = lambda x: ((1 + math.cos(x * math.pi / config['epochs'])) / 2) * (1 - config['lrf']) + config['lrf']  # cosine
#     scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
#
#     for epoch in range(config['epochs']):
#         # train
#         train_loss, train_acc = train_one_epoch(model=model,
#                                                 optimizer=optimizer,
#                                                 data_loader=train_loader,
#                                                 device=device,
#                                                 epoch=epoch)
#
#         scheduler.step()
#
#         # validate
#         val_loss, val_acc = evaluate(model=model,
#                                      data_loader=val_loader,
#                                      device=device,
#                                      epoch=epoch)
#
#         tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
#         tb_writer.add_scalar(tags[0], train_loss, epoch)
#         tb_writer.add_scalar(tags[1], train_acc, epoch)
#         tb_writer.add_scalar(tags[2], val_loss, epoch)
#         tb_writer.add_scalar(tags[3], val_acc, epoch)
#         tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
#
#         torch.save(model.state_dict(), "../data/model_save/model-{}.pth".format(epoch))
