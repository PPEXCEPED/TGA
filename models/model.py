import torch.nn
import torch.nn.functional as F
import torch.nn as nn
import process.ImageFeature as ImageFeature
import process.TextFeatureEmbedding as TextFeature
from process import FuseAllFeature
from process.DataSet import *
from torch.autograd import Variable, Function


class ReverseLayerF(Function):
    def forward(self, x):
        self.lambd = 1
        return x.virw_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grade_reverse(x):
    return ReverseLayerF.apply(x)


class FeatureExtractor(torch.nn.Module):
    def __init__(self, lstm_dropout_rate):
        super(FeatureExtractor, self).__init__()
        self.text = TextFeature.ExtractTextFeature(TEXT_LENGTH)
        # self.text_fuse = FuseAllFeature.ModalityFusion_1(512)
        self.image = ImageFeature.ExtractImageFeature()
        self.image_fuse = FuseAllFeature.ModalityFusion_1(512)

    def forward(self, text_index, image_feature):
        text_seq = self.text(text_index)
        # text_fusion = self.text_fuse(text_result, text_seq.permute(1, 0, 2))
        image_result, image_seq = self.image(image_feature)
        # print(image_result.shape, image_seq.shape)
        image_fusion = self.image_fuse(image_result, image_seq)
        return text_seq, image_fusion


class Siamese(torch.nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.siamese_text = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128))
        self.siamese_image = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128))

    def forward(self, text_feature=None, image_feature=None):
        output1 = self.siamese_text(text_feature)
        output2 = self.siamese_image(image_feature)
        return output1, output2


class LabelClassfier(torch.nn.Module):
    def __init__(self):
        super(LabelClassfier, self).__init__()
        self.label_classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
            nn.Sigmoid())

    def forward(self, feature):
        label_pred = self.label_classifier(feature)
        return label_pred


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
