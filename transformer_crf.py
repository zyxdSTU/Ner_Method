from dataLoader import tagDict
from dataLoader import tag2int
from dataLoader import int2tag
from dataLoader import wordDict
from dataLoader import word2int
from dataLoader import int2word
from tqdm import tgrange
from tqdm import tqdm
from seqeval.metrics import f1_score, accuracy_score, classification_report
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import torch.nn
class Transformer_CRF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wordDictSize = len(wordDict)
        self.embeddingSize = config['model']['embeddingSize']
        self.dropout = nn.Dropout(config['model']['dropout'])
        self.hiddenSize = config['model']['hiddenSize']

        position  = PositionalEncoding(self.embeddingSize, dropout = 0.1)
        self.wordEmbeddings = nn.Sequential(Embeddings(self.embeddingSize, self.wordDictSize), position)

        #self.wordEmbeddings = nn.Embedding(self.wordDictSize, self.embeddingSize)

        self.layer = nn.TransformerEncoderLayer(d_model = self.embeddingSize, nhead = 4)

        self.encoder = nn.TransformerEncoder(self.layer, num_layers=3)

        self.lstm = nn.LSTM(input_size=self.embeddingSize, hidden_size= self.hiddenSize // 2, batch_first=True, 
        bidirectional=True, num_layers=1)

        self.fc = nn.Linear(self.embeddingSize, len(tagDict))

    def forward(self, batchSentence):
        if self.training:
            #获得词嵌入
            mask = batchSentence == 0
            embeds = self.wordEmbeddings(batchSentence)
            embeds = embeds.permute(1, 0, 2)
            encoderFeature = self.encoder(embeds, src_key_padding_mask=mask)
            encoderFeature = encoderFeature.permute(1, 0, 2)
            encoderFeature, _ = self.lstm(encoderFeature)
            tagFeature = self.fc(encoderFeature)
            tagFeature = self.dropout(tagFeature)
            tagScores = F.log_softmax(tagFeature, dim=2)
        else:
            with torch.no_grad():
                mask = batchSentence.data == 0
                embeds = self.wordEmbeddings(batchSentence)
                embeds = embeds.permute(1, 0, 2)
                encoderFeature = self.encoder(embeds, src_key_padding_mask=mask)
                encoderFeature = encoderFeature.permute(1, 0, 2)
                encoderFeature, _ = self.lstm(encoderFeature)
                tagFeature = self.fc(encoderFeature)
                tagScores = F.log_softmax(tagFeature, dim=2)
        return tagScores

def transformerCRFTrain(net, iterData, optimizer, criterion, DEVICE):
    net.train()
    totalLoss, number = 0, 0
    for batchSentence, batchTag, lenList in tqdm(iterData):
        batchSentence = batchSentence.to(DEVICE)
        batchTag = batchTag.to(DEVICE)
        net.zero_grad()
        tagScores  = net(batchSentence)

        loss = 0
        for index, element in enumerate(lenList):
            tagScore = tagScores[index][:element]
            tag = batchTag[index][:element]
            loss +=  criterion(tagScore, tag)

        loss.backward()
        optimizer.step()
        totalLoss += loss.item(); number += 1
    return totalLoss / number

def transformerCRFEval(net, iterData, criterion, DEVICE):
    net.eval()
    totalLoss, number = 0, 0
    yTrue, yPre, ySentence = [], [], []
    for batchSentence, batchTag, lenList in tqdm(iterData):
        batchSentence = batchSentence.to(DEVICE)
        batchTag = batchTag.to(DEVICE)
        net.zero_grad()
        tagScores  = net(batchSentence)

        loss = 0
        for index, element in enumerate(lenList):
            tagScore = tagScores[index][:element]
            tag = batchTag[index][:element]
            sentence = batchSentence[index][:element]
            loss +=  criterion(tagScore, tag)
            yTrue.append(tag.cpu().numpy().tolist())
            ySentence.append(sentence.cpu().numpy().tolist())
            yPre.append([element.argmax().item() for element in tagScore])

        number = number + 1
        totalLoss += loss.item()

    yTrue2tag = [[int2tag[element2] for element2 in element1] for element1 in yTrue]
    yPre2tag = [[int2tag[element2] for element2 in element1] for element1 in yPre]
    f1Score = f1_score(y_true=yTrue2tag, y_pred=yPre2tag)

    return totalLoss / number, f1Score
  
#词嵌入
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                                -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                            requires_grad=False)
        return self.dropout(x)