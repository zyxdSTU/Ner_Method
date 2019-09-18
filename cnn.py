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

class CNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wordDictSize = len(wordDict)
        self.embeddingSize = config['model']['embeddingSize']
        self.hiddenSize = config['model']['hiddenSize']
        self.wordEmbeddings = nn.Embedding(self.wordDictSize, self.embeddingSize)

        #选取的特征数量
        self.featureLen = config['model']['featureLen']

        self.cnnArr = [(nn.Conv2d(in_channels=1, out_channels=self.hiddenSize/self.featureLen, kernel_size=(i, self.embeddingSize)), i)
            for i in range(2, 2+featureLen)]

        self.fc = nn.Linear(self.hiddenSize, len(tagDict))

    def forward(self, batchSentence):

        embeddings = self.wordEmbeddings(batchSentence)

        result = []
        for cnn, size in cnnArr:
            #左、右边padding
            if size % 2 != 0:
                paddingLef, paddingRig = (size - 1) / 2
            else:
                paddingLef, paddingRig = size / 2 , size / 2 -1
            
            paddingLef = torch.zeros((self.embeddingSize, paddingLef)).long()
            paddingRig = torch.zeros((self.embeddingSize, paddingRig)).long()

            inputData = torch.cat((paddingLef,embeddings, paddingRig), 1)
            print (inputData.shape)
            inputData.unsqueeze(1)
            outputData = cnn(inputData)
            outputData = outputData.squeeze().transpose(1, 2)
            print (inputData.shape)
            result.append(outputData)
        
        result = torch.cat(result, 2)

        result = self.cf(result)

        return result

def cnnTrain(net, iterData, optimizer, criterion, DEVICE):
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

def cnnEval(net, iterData, criterion, DEVICE):
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





