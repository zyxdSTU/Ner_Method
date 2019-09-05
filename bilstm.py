from dataLoader import tagDict
from dataLoader import tag2int
from dataLoader import int2tag
from dataLoader import wordDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tgrange
from tqdm import tqdm
from seqeval.metrics import f1_score, accuracy_score, classification_report

class BiLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wordDictSize = len(wordDict)
        self.embeddingSize = config['model']['embeddingSize']
        self.hiddenSize = config['model']['hiddenSize']
        self.dropout = nn.Dropout(config['model']['dropout'])
        self.wordEmbeddings = nn.Embedding(self.wordDictSize, self.embeddingSize)

        self.lstm = nn.LSTM(input_size=self.embeddingSize, hidden_size= self.hiddenSize // 2, batch_first=True, bidirectional=True, num_layers=2)
        self.fc = nn.Linear(self.hiddenSize, len(tagDict))
    
    def forward(self, batchSentence):
        if self.training:
            #获得词嵌入
            embeds = self.wordEmbeddings(batchSentence)
            lstmFeature,_ = self.lstm(embeds)
            tagFeature = self.fc(lstmFeature)
            lstmFeature = self.dropout(lstmFeature)
            tagScores = F.log_softmax(tagFeature, dim=2)
        else:
            with torch.no_grad():
                embeds = self.wordEmbeddings(batchSentence)
                lstmFeature, _ = self.lstm(embeds)    
                tagFeature = self.fc(lstmFeature)
                tagScores = F.log_softmax(tagFeature, dim=2)
        return tagScores

def bilstmTrain(net, iterData, optimizer, criterion, DEVICE):
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
        number = number + 1
        totalLoss += loss.item()
    return totalLoss / number

def bilstmEval(net, iterData, criterion, DEVICE):
    net.eval()
    totalLoss, number = 0, 0
    yTrue, yPre, ySentence = [], [], []
    for batchSentence, batchTag, lenList in tqdm(iterData):
        batchSentence = batchSentence.to(DEVICE)
        batchTag = batchTag.to(DEVICE)
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
  
