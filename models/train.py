import time

import torch

from process import ImageFeature, FuseAllFeature, FinalClassifier, TextFeatureEmbedding
from process.DataSet import *
from torch.utils.data import Dataset, DataLoader
import os
import model
import torch.nn.functional as F

from tools.config import config
from models.model import Siamese, ContrastiveLoss, FeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# loss = Visdom()
# acc = Visdom()
# loss.line([[0, 0]], [1], win='train_loss', opts=dict(title='loss', legend=['siamese_loss', 'domain_loss']))
# acc.line([[0, 0]], [1], win='train_acc', opts=dict(title='acc', legend=['siamese_acc', 'domain_acc']))
TEXT_HIDDEN = 256

def to_np(x):
    return x.data.cpu().numpy()


def create_folder(foldermodal_names):
    current_position = "./model_sava/"
    foldermodal_name = str(current_position) + str(foldermodal_names) + "/"
    isCreate = os.path.exists(foldermodal_name)
    if not isCreate:
        os.makedirs(foldermodal_name)
        print(str(foldermodal_name) + 'is created')
    else:
        print('Already exist')
        return False


# def model_load():
#     checkpoint1 = torch.load('../data/model_sava/extractor.pth')
#     checkpoint2 = torch.load('../data/model_sava/siamese.pth')
#     extractor.load_state_dict(checkpoint1['model'])
#     siamese.load_state_dict(checkpoint2['model'])
#     optimizer_extractor.load_state_dict(checkpoint1['optimizer'])
#     optimizer_siamese.load_state_dict(checkpoint2['optimizer'])
#     print('Model_Load completed\n')
#     return extractor, siamese, optimizer_extractor, optimizer_siamese


class Multimodel_image_text_g(torch.nn.Module):
    def __init__(self, lstm_dropout_rate, fc_dropout_rate):
        super(Multimodel_image_text_g, self).__init__()
        self.image = ImageFeature.ExtractImageFeature()
        # transformer 的参数
        # self.text = TextFeatureEmbedding.ExtractTextFeature(TEXT_LENGTH, lstm_dropout_rate)
        # LSTM
        self.text = TextFeatureEmbedding.ExtractTextFeature(TEXT_LENGTH, lstm_dropout_rate)
        # 基于注意力机制的拼接
        self.fuse = FuseAllFeature.ModalityFusion_2(1024, 256)
        # 直接拼接
        # self.fuse = FuseAllFeature.ModalityFusion_concat(1024, 256)
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate, 512)

    def forward(self, image_feature, text_index):
        image_result, image_seq = self.image(image_feature, text_index)
        text_result, text_seq = self.text(text_index)
        fusion = self.fuse(image_result, image_seq, text_result, text_seq.permute(1, 0, 2))
        output = self.final_classifier(fusion)
        return output


class text_g(torch.nn.Module):
    def __init__(self, lstm_dropout_rate, fc_dropout_rate):
        super(text_g, self).__init__()
        # self.image = ImageFeature.ExtractImageFeature()
        self.text = TextFeatureEmbedding.ExtractTextFeature(TEXT_LENGTH, lstm_dropout_rate)
        self.fuse = FuseAllFeature.ModalityFusion_1(256)
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate, 512)

    def forward(self, image_feature, text_index):
        # image_result, image_seq = self.image(image_feature)
        text_result, text_seq = self.text(text_index)
        fusion = self.fuse(text_result, text_seq.permute(1, 0, 2))
        output = self.final_classifier(fusion)
        return output


class image_g(torch.nn.Module):
    def __init__(self, lstm_dropout_rate, fc_dropout_rate):
        super(image_g, self).__init__()
        self.image = ImageFeature.ExtractImageFeature()
        # self.text = TextFeatureEmbedding.ExtractTextFeature(TEXT_LENGTH, lstm_dropout_rate)
        self.fuse = FuseAllFeature.ModalityFusion_1(1024)
        self.final_classifier = FinalClassifier.ClassificationLayer(fc_dropout_rate, 512)

    def forward(self, image_feature, text_index):
        image_result, image_seq = self.image(image_feature)
        # text_result, text_seq = self.text(text_index)
        fusion = self.fuse(image_result, image_seq)
        output = self.final_classifier(fusion)
        return output


def train(model5, siamese, extractor, train_loader, val_loader, loss_fn, siamese_loss_fn, optimizer5, optimizer_siamese,
          number_of_epoch, i, file):
    F1_old = 0
    TP = TN = FN = FP = 0
    F1_5 = 0
    TP5 = TN5 = FN5 = FP5 = 0
    p5 = r5 = 0
    for epoch in range(number_of_epoch):

        model5_train_loss = 0
        sim_train_loss = 0
        sim_correct_train = 0
        # for i in range(1,9):
        #     model_train_loss[i]=0
        #     model_correct_train[i] = 0
        #     eval('model'+str(i)+'.train()')
        model5_correct_train = 0
        model5.train()
        # siamese.train()
        # extractor.train()

        right_num = data_num = count = 0
        dict = {}
        for text_index, image_feature, group, id in train_loader:
            count += 1
            dataset_len = len(train_loader)
            data_num += train_loader.batch_size
            group = group.view(-1, 1).to(torch.float32).to(device)
            model5_pred = model5(image_feature.to(device), text_index.to(device))
            # print("image_feature:{}".format(image_feature.shape))
            text_feature, image_feature1 = extractor(text_index.to(device), image_feature.to(device))
            # # print("image_feature1:{}".format(image_feature1.shape))
            output1, output2 = siamese(text_feature, image_feature1)
            distance = F.pairwise_distance(output1, output2)

            for j in range(len(distance)):
                if distance[j] > 0.65:
                    model5_pred[j] = model5_pred[j] + 0.1 * distance[j]
            # print(distance, model5_pred)
            # exit(0)

            model5_loss = loss_fn(model5_pred, group)
            # print(model5_loss)
            model5_train_loss += model5_loss

            model5_correct_train += (model5_pred.round() == group).sum().item()

            optimizer5.zero_grad()

            model5_loss.backward()
            # sim_loss.backward(retain_graph=True)

            optimizer5.step()
            # optimizer_siamese.step()

            def f1_score(TP, TN, FN, FP, model):
                TP += ((model.round() == 1) & (group == 1)).cpu().sum().item()
                TN += ((model.round() == 0) & (group == 0)).cpu().sum().item()
                FN += ((model.round() == 0) & (group == 1)).cpu().sum().item()
                FP += ((model.round() == 1) & (group == 0)).cpu().sum().item()
                try:
                    p = TP / (TP + FP)
                    r = TP / (TP + FN)
                    F1 = 2 * r * p / (r + p)
                except:
                    pass
                return TP, TN, FN, FP, p, r, F1

            try:
                TP5, TN5, FN5, FP5, p5, r5, F1_5 = f1_score(TP5, TN5, FN5, FP5, model5_pred)
            except:
                pass
                # try:
            #     # compute F1 precision准确率 recall召回率
            #     p = TP / (TP + FP)
            #     # p = torch.floor_divide(TP,TP+FP)
            #     r = TP / (TP + FN)
            #     # r = torch.floor_divide(TP, TP + FN)
            #     F1 = 2 * r * p / (r + p)
            # except:
            #     pass
            try:
                print('Training...epoch:%d/%d' % (epoch + 1, number_of_epoch))
                print('batch:%d/%d' % (count, dataset_len))
                print(
                    "pic_txt_guid:train_loss=%.5f train_acc=%.3f precision=%.5f recall=%.5f F1_score=%.5f" % (
                        model5_train_loss / data_num, model5_correct_train / data_num, p5, r5, F1_5))
                print('')
            except:
                print('Training...epoch:%d/%d' % (epoch + 1, number_of_epoch))
                print('batch:%d/%d' % (count, dataset_len))

                print("pic_txt_guid:train_loss=%.5f train_acc=%.3f " % (
                    model5_train_loss / data_num, model5_correct_train / data_num))
                print('')

        file.write(
            'pic_txt_guid:train_loss={} train_acc={} precision={} recall={} F1_score={} '.format(
                model5_train_loss / data_num,
                model5_correct_train / data_num,
                p5, r5, F1_5))
        file.write('\n')
        # file.close()

        # #chart
        # 			loss.line([[model1_train_loss.item()/data_num,model2_train_loss.item()/data_num
        # 				,model3_train_loss.item()/data_num,model4_train_loss.item()/data_num,model5_train_loss.item()/data_num
        # 				,model6_train_loss.item()/data_num,model7_train_loss.item()/data_num,model8_train_loss.item()/data_num]],
        # 				[count], win='train_loss', update='append')
        # 			acc.line([[model1_correct_train/data_num,model2_correct_train/data_num
        # 				,model3_correct_train/data_num,model4_correct_train/data_num,model5_correct_train/data_num
        # 				,model6_correct_train/data_num,model7_correct_train/data_num,model8_correct_train/data_num,right_num/data_num]],
        # 				[count], win='train_acc', update='append')

        # learning_rate adjustment
        F1_new = test(model5, extractor, siamese, val_loader, file)

        # test(model1, model2, model3,  val_loader)
        if F1_new < F1_old:
            print('Learning rate changed')

            optimizer5.param_groups[0]['lr'] *= 0.8

        F1_old = F1_new

        print('learning_rate:', config['lr'])
        print('F1:', F1_new)
        print('')

        # sava model

        state5 = {'model': model5.state_dict(), 'optimizer': optimizer5.state_dict()}

        name = 'model' + str(i) + "/"

        torch.save(state5, './model_sava/' + name + 'model5.pth')


def test(model5, extractor, siamese, val_loader, file):
    valid_loss5 = 0

    correct_valid5 = 0

    model5.eval()
    right_num = data_num = count = 0

    F1 = 0

    TP5 = TN5 = FN5 = FP5 = 0
    F1_5 = 0
    p5 = r5 = 0
    with torch.no_grad():
        for val_text_index, val_image_feature, val_grad, val_id in val_loader:
            count += 1
            dataset_len = len(val_loader)
            val_group = val_grad.view(-1, 1).to(torch.float32).to(device)

            model5_val_pred = model5(val_image_feature.to(device), val_text_index.to(device))
            text_feature, image_feature1 = extractor(val_text_index.to(device), val_image_feature.to(device))
            # # print("image_feature1:{}".format(image_feature1.shape))
            output1, output2 = siamese(text_feature, image_feature1)

            data_num += val_loader.batch_size

            val_loss5 = loss_fn(model5_val_pred, val_group)

            distance = F.pairwise_distance(output1, output2)
            for j in range(len(distance)):
                if distance[j] > 0.65:
                    model5_val_pred[j] = model5_val_pred[j] + 0.1 * distance[j]

            valid_loss5 += val_loss5

            # valid_loss5 = valid_loss5 + 0.25 * distance

            correct_valid5 += (model5_val_pred.round() == val_group).sum().item()

            def f1_score(TP, TN, FN, FP, model):
                p = r = F1 = 0
                TP += ((model.round() == 1) & (val_group == 1)).cpu().sum().item()
                TN += ((model.round() == 0) & (val_group == 0)).cpu().sum().item()
                FN += ((model.round() == 0) & (val_group == 1)).cpu().sum().item()
                FP += ((model.round() == 1) & (val_group == 0)).cpu().sum().item()
                try:
                    p = TP / (TP + FP)
                    r = TP / (TP + FN)
                    F1 = 2 * r * p / (r + p)
                    return TP, TN, FN, FP, p, r, F1
                except:
                    return TP, TN, FN, FP, p, r, F1
                # return TP, TN, FN, FP, p, r, F1

            # try:

            TP5, TN5, FN5, FP5, p5, r5, F1_5 = f1_score(TP5, TN5, FN5, FP5, model5_val_pred)

            # except:
            #     pass
            try:
                print('Validing...')
                print('batch:%d/%d' % (count, dataset_len))

                print(
                    "pic_txt_guid:train_loss=%.5f train_acc=%.3f precision=%.5f recall=%.5f F1_score=%.5f" % (
                        valid_loss5 / data_num, correct_valid5 / data_num, p5, r5, F1_5))
                print('')
            except:
                print('Validing...')
                print('batch:%d/%d' % (count, dataset_len))

                print("pic_txt_guid:train_loss=%.5f train_acc=%.3f " % (
                    valid_loss5 / data_num, correct_valid5 / data_num))
                print('')

    file.write(
        'v_pic_txt_guid:train_loss={} train_acc={} precision={} recall={} F1_score={} '.format(
            valid_loss5 / data_num,
            correct_valid5 / data_num,
            p5, r5, F1_5))
    file.write('\n')
    return F1


def model_load(name):
    checkpoint5 = torch.load('./model_sava/' + name + 'model5.pth')
    model5.load_state_dict(checkpoint5['model'])
    optimizer5.load_state_dict(checkpoint5['optimizer'])

    print('Model_Load completed\n')
    return model5, optimizer5


learning_rate_list = [0.001]
fc_dropout_rate_list = [0, 0.3, 0.9, 0.99]
lstm_dropout_rate_list = [0, 0.2, 0.4]
weight_decay_list = [0, 1e-6, 1e-5, 1e-4]

learning_rate = 0.001
lstm_dropout_rate = 0
weight_decay = 0
train_fraction = 0.9
val_fraction = 0.1
batch_size = 32
data_shuffle = True
number_of_epoch = 5

train_data = train_data_set
test_set = test_data_set
train_set, val_set = train_val_split(train_data, config['val_ratio'])
print(f'train_set_length:{len(train_set)} val_set_length:{len(val_set)}')
# load data
train_loader = train_loader
val_loader = val_loader
test_loader = test_loader

# all_Data = RumorDataset(train_data_set)
# train_fraction = 0.8
# val_fraction = 0.2
# train_set, val_set, test_set = train_val_split(all_Data, train_fraction, val_fraction)
# batch_size = 24
# data_shuffle = True
#
# # load data
#
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=data_shuffle)
# val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=data_shuffle)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=data_shuffle)
# play_loader = DataLoader(test_set, batch_size=1, shuffle=data_shuffle)

# start train
import itertools

if __name__ == "__main__":
    comb = itertools.product(learning_rate_list, fc_dropout_rate_list, lstm_dropout_rate_list, weight_decay_list)
    i = 1
    for learning_rate, fc_dropout_rate, lstm_dropout_rate, weight_decay in list(comb)[10:]:
        print(
            f"learning rate={learning_rate} | fc dropout={fc_dropout_rate} | lstm dropout={lstm_dropout_rate} | weight decay={weight_decay}")
        # loss function
        loss_fn = torch.nn.BCELoss()
        siamese_loss_fn = ContrastiveLoss()
        # initilize the model

        model5 = Multimodel_image_text_g(lstm_dropout_rate, fc_dropout_rate).to(device)
        # 单模态 文本
        # model5 = text_g(lstm_dropout_rate, fc_dropout_rate).to(device)
        # 单模态 图像
        # model5 = image_g(lstm_dropout_rate, fc_dropout_rate).to(device)
        # optimizer
        siamese = Siamese().to(device)
        extractor = FeatureExtractor(lstm_dropout_rate).to(device)

        optimizer_siamese = torch.optim.Adam(siamese.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer5 = torch.optim.Adam(model5.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer_extractor = torch.optim.Adam(extractor.parameters(), lr=learning_rate, weight_decay=weight_decay)
        name = 'model' + str(i) + "/"
        create_folder(name)
        file = open('./model_sava/' + name + 'r.txt', 'a')
        file.write(
            f"learning rate={learning_rate} | fc dropout={fc_dropout_rate} | lstm dropout={lstm_dropout_rate} | weight decay={weight_decay}")
        if os.path.exists('./model_sava/' + name + 'model1.pth'):
            model_load(name)

        # train
        number_of_epoch = 15
        time_start = time.time()
        # print(i)
        train(model5, siamese, extractor, train_loader, val_loader, loss_fn, siamese_loss_fn,
              optimizer5, optimizer_siamese,
              number_of_epoch, i, file)

        i += 1
        time_end = time.time()
        print('time cost', (time_end - time_start) / 60, 'min')
        print(
            f"learning rate={learning_rate} | fc dropout={fc_dropout_rate} | lstm dropout={lstm_dropout_rate} | weight decay={weight_decay}")

        # file.close

        input('将模型放入对应文件夹后按回车继续...\n')

### FNPS
# def train(extractor, siamese, optimizer_extractor, optimizer_siamese,
#           siamese_loss_fn, train_loader, number_of_epoch, is_train):
#     step = 0
#     for epoch in range(number_of_epoch):
#         siamese_loss_sum = siamese_correct_num = data_num = 0
#         extractor.train()
#         siamese.train()
#         batch = 0
#         for text_index, image_feature, label, id in train_loader:
#             step += 1
#             batch += 1
#             batch_num = len(train_loader)
#             data_num += label.shape[0]
#             label = label.view(-1, 1).to(torch.float32).to(device)
#
#             text_feature, image_feature = extractor(text_index.to(device), image_feature.to(device))
#             print('text_feature shape:{}, image_feature shape:{}'.format(text_feature.shape, image_feature.shape))
#
#             ########### 上面提取完文本特征和图像特征，应该使用注意力机制融合，再进行谣言检测
#
#             output1, output2 = siamese(text_feature, image_feature)
#
#             # print('output1={}'.format(output1), 'output2={}'.format(output2))
#             distance = F.pairwise_distance(output1, output2)
#             one = torch.ones_like(distance)
#             zero = torch.zeros_like(distance)
#             siamese_pred = torch.where(distance <= 0.65, zero, one).round().view(-1, 1)
#
#             # print(siamese_pred.shape)
#             siamese_loss = siamese_loss_fn(output1, output2, label)
#             siamese_loss_sum += siamese_loss.item()
#             siamese_correct_num += (siamese_pred == label).sum().item()
#
#             optimizer_extractor.zero_grad()
#             optimizer_siamese.zero_grad()
#
#             siamese_loss.backward(retain_graph=True)
#
#             if is_train == 1:
#                 optimizer_siamese.step()
#                 optimizer_extractor.step()
#             else:
#                 pass
#
#             print('Training...epoch:%d/%d' % (epoch + 1, number_of_epoch))
#             print('batch:%d/%d' % (batch, batch_num))
#             print("siamese:train_loss=%.5f train_acc=%.4f" % (
#                 siamese_loss_sum / data_num, siamese_correct_num / data_num))
#
#             print('')
#             # chart
#             # loss.line([[siamese_loss_sum / data_num]],
#             #           [step], win='train_loss', update='append')
#             # acc.line([[siamese_correct_num / data_num]],
#             #          [step], win='train_acc', update='append')
#
#             if batch == 1:
#                 lable_list = list(to_np(label.squeeze()))
#                 pred_list = list(to_np(siamese_pred.round().squeeze()))
#             else:
#                 lable_list = lable_list + list(to_np(label.squeeze()))
#                 pred_list = pred_list + list(to_np(siamese_pred.round().squeeze()))
#
#         precision = precision_score(lable_list, pred_list)
#         recall = recall_score(lable_list, pred_list)
#         f1 = f1_score(lable_list, pred_list)
#         print('Precision：' + str(precision))
#         print('Recall：' + str(recall))
#         print('F1-score：' + str(f1))
#         # sava model
#         state1 = {'model': extractor.state_dict(), 'optimizer': optimizer_extractor.state_dict()}
#         state2 = {'model': siamese.state_dict(), 'optimizer': optimizer_siamese.state_dict()}
#         torch.save(state1, './model_sava/extractor.pth')
#         torch.save(state2, './model_sava/siamese.pth')
#
#
# learning_rate = 0.001
# lstm_dropout_rate = 0
# weight_decay = 0
# train_fraction = 0.9
# val_fraction = 0.1
# batch_size = 64  #############
# data_shuffle = True
# number_of_epoch = 5  #############
# # train_data = RumorDataset(train_data_set)
# # test_set = RumorDataset(test_data_set)
# train_data = train_data_set
# test_set = test_data_set
# train_set, val_set = train_val_split(train_data, config['val_ratio'])
# print(f'train_set_length:{len(train_set)} val_set_length:{len(val_set)}')
# # load data
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=data_shuffle)
# val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=data_shuffle)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=data_shuffle)
# # start train
# if __name__ == "__main__":
#     # loss function
#     siamese_loss_fn = model.ContrastiveLoss()
#     domain_loss_fn = torch.nn.CrossEntropyLoss()
#     # initilize the model
#     extractor = model.FeatureExtractor(lstm_dropout_rate).to(device)
#     siamese = model.Siamese().to(device)
#
#     # optimizer
#     optimizer_extractor = torch.optim.Adam(extractor.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     optimizer_siamese = torch.optim.Adam(siamese.parameters(), lr=learning_rate, weight_decay=weight_decay)
#     # model load
#     modal_path = '../data/model_sava/'
#     if os.listdir('../data/model_sava/') != []:
#         model_load()
#     # train
#     train(extractor, siamese, optimizer_extractor, optimizer_siamese,
#           siamese_loss_fn, train_loader, number_of_epoch, 11)
