import math
import numpy as np
import torch
from torch import nn
import process.DataSet as DataSet

## 词嵌入 文本=》向量

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


device = get_device()


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class ExtractTextFeature(nn.Module):
    def __init__(self, text_length, dropout=0.2):
        super(ExtractTextFeature, self).__init__()
        self.text_length = text_length
        self.embedding_weight = self.getEmbedding() # [63907, 32]
        self.embedding_size = self.embedding_weight.shape[1]
        self.embedding = nn.Embedding.from_pretrained(self.embedding_weight)

        self.position_embedding = PositionalEncoding(32, dropout)
        encoder_layer = nn.TransformerEncoderLayer(32, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.linear = nn.Linear(32, DataSet.TEXT_HIDDEN)

    def getEmbedding(self):
        return torch.from_numpy(np.loadtxt("../data/text_embedding/vectors.txt", delimiter=' ', dtype='float32'))

    def forward(self, input):
        input = input.long()
        input = self.embedding(input)
        # print(input.shape)
        input = self.position_embedding(input)
        # print(input.shape) #torch.Size([32, 75, 32])
        # print(input)
        out = self.transformer_encoder(input)
        # print(out.shape) #torch.Size([32, 75, 32])
        # print(out)
        out = self.linear(out)
        state = torch.mean(out, 1)
        return state, out


if __name__ == '__main__':
    test = ExtractTextFeature(DataSet.TEXT_LENGTH)
    for text_index, image_feature, label, id in DataSet.train_loader:
        # text_index([32, 75])
        seq, out = test(text_index)
        print(seq.shape) # [32, 256]
        break

# import torch
# import numpy as np
#
# from process import DataSet
#
#
# class ExtractTextFeature(torch.nn.Module):
#     def __init__(self,text_length,hidden_size,dropout_rate=0.2):
#         super(ExtractTextFeature, self).__init__()
#         self.hidden_size=hidden_size   #256
#         self.text_length=text_length   #75
#         embedding_weight=self.getEmbedding()  #torch.Size([12280, 200])
#         self.embedding_size=embedding_weight.shape[1]   #200
#         self.embedding=torch.nn.Embedding.from_pretrained(embedding_weight)
#         self.biLSTM=torch.nn.LSTM(input_size=32,hidden_size=hidden_size,bidirectional=True,batch_first=True)
#         # early fusion
#         self.Linear_1=torch.nn.Linear(32,hidden_size)
#         self.Linear_2=torch.nn.Linear(32,hidden_size)
#         self.Linear_3=torch.nn.Linear(32,hidden_size)
#         self.Linear_4=torch.nn.Linear(32,hidden_size)
#         self.linear = torch.nn.Linear(512, DataSet.TEXT_HIDDEN)
#
#         # dropout
#         self.dropout=torch.nn.Dropout(dropout_rate)
#
#     def forward(self, input):  #input:[32,75]
#         embedded=self.embedding(input).view(-1, self.text_length, self.embedding_size)
#
#         # if(guidence is not None):
#         #     # early fusion
#         #     hidden_init=torch.stack([torch.relu(self.Linear_1(guidence)),torch.relu(self.Linear_2(guidence))],dim=0)
#         #     cell_init=torch.stack([torch.relu(self.Linear_3(guidence)),torch.relu(self.Linear_4(guidence))],dim=0)
#         #     output,_=self.biLSTM(embedded,(hidden_init,cell_init))
#         # else:
#         output,_=self.biLSTM(embedded,None)
#         output = self.linear(output)
#         # dropout
#         # output=self.dropout(output)
#
#         RNN_state=torch.mean(output,1)
#         return RNN_state,output
#
#     def getEmbedding(self):
#         return torch.from_numpy(np.loadtxt("../data/text_embedding/vectors.txt", delimiter=' ', dtype='float32'))
#
#
# if __name__ == "__main__":
#     test=ExtractTextFeature(DataSet.TEXT_LENGTH, DataSet.TEXT_HIDDEN)
#     for text_index, image_feature, label, id in DataSet.train_loader:
#         result,seq=test(text_index)
#         print(result.shape)#torch.Size([32, 512])
#         # print(seq)#torch.Size([32, 75, 512])
#         break
