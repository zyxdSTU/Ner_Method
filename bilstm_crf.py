import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.rnn as rnn
import operator
from dataLoader import tagDict
from dataLoader import int2tag
from dataLoader import tag2int
from dataLoader import wordDict
from dataLoader import int2word
from dataLoader import word2int
from tqdm import tgrange
from tqdm import tqdm
from seqeval.metrics import f1_score, accuracy_score, classification_report

START_TAG = '<START>'
STOP_TAG = '<STOP>'
PAD = '<PAD>'

#计算log sum exp
def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))

class BiLSTM_CRF(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddingSize = config['model']['embeddingSize']
        self.hiddenSize = config['model']['hiddenSize']
            
        self.DEVICE = config['DEVICE']
        self.dropout = nn.Dropout(config['model']['dropout'])

        self.wordEmbeddings = nn.Embedding(self.wordDictSize, self.embeddingSize)

        self.tagDict = tagDict.copy();  self.tagDict.extend(['<START>', '<STOP>'])

        self.lstm = nn.LSTM(input_size=self.embeddingSize, hidden_size= self.hiddenSize // 2, batch_first=True, 
        bidirectional=True, num_layers=2)

        self.fc = nn.Linear(self.hiddenSize, len(self.tagDict))

        self.crf = crf(self.tagDict, config)

    def forward(self, batchSentence, batchTag):
        mask = batchSentence.data.gt(0).float()

        embeds = self.wordEmbeddings(batchSentence)
        h, _ = self.lstm(embeds)

        h = self.fc(h)

        #使用dropout策略
        h = self.dropout(h)

        h *= mask.unsqueeze(2)
        forwardScore = self.crf.forward(h, mask)
        goldScore = self.crf.score(h, batchTag, mask)

        return torch.mean(forwardScore - goldScore)
    
    def decode(self, batchSentence):
        mask = batchSentence.data.gt(0).float()

        #bert embedding
        embeds = self.wordEmbeddings(batchSentence)
        h, _ = self.lstm(embeds)
        h = self.fc(h)
        h *= mask.unsqueeze(2)
        return self.crf.decode(h, mask)

def bilstmCRFTrain(net, iterData, optimizer, criterion, DEVICE):
    net.train()
    totalLoss, number = 0, 0
    for batchSentence, batchTag, lenList in tqdm(iterData):
        batchSentence = batchSentence.to(DEVICE)
        batchTag = batchTag.to(DEVICE)
        net.zero_grad()
        loss = net(batchSentence, batchTag)
        loss.backward()
        optimizer.step()

        totalLoss += loss.item(); number += 1
        
    return totalLoss / number

def bilstmCRFEval(net, iterData, criterion, DEVICE):
    net.eval()
    totalLoss, number = 0, 0
    yTrue, yPre, ySentence = [], [], []
    for batchSentence, batchTag, lenList in tqdm(iterData):

        batchSentence = batchSentence.to(DEVICE)
        batchTag = batchTag.to(DEVICE)
        
        loss = net(batchSentence, batchTag)
        tagPre  = net.decode(batchSentence)

        tagTrue = [element[:length] for element, length in zip(batchTag.cpu().numpy(), lenList)]
        sentence = [element[:length] for element, length in zip(batchSentence.cpu().numpy(), lenList)]
        yTrue.extend(tagTrue); yPre.extend(tagPre); ySentence.extend(sentence)
        
        totalLoss += loss.item(); number += 1

    yTrue2tag = [[int2tag[int(element2)] for element2 in element1] for element1 in yTrue]
    yPre2tag = [[int2tag[int(element2)] for element2 in element1] for element1 in yPre]

    f1Score = f1_score(y_true=yTrue2tag, y_pred=yPre2tag)

    return totalLoss / number, f1Score
  
class crf(nn.Module):
    def __init__(self, tagDict, config):
        super().__init__()

        self.tagDict = tagDict
        self.config = config
        self.DEVICE = self.config['DEVICE']
        self.tag2int ={element:index for index, element in enumerate(self.tagDict)}
        self.int2tag ={index:element for index, element in enumerate(self.tagDict)}
        self.num_tags = len(self.tagDict)

        # matrix of transition scores from j to i
        self.trans = nn.Parameter(torch.randn(self.num_tags, self.num_tags))
        self.trans.data[self.tag2int[START_TAG], :] = -10000. # no transition to SOS
        self.trans.data[:, self.tag2int[STOP_TAG]] = -10000. # no transition from EOS except to PAD
        self.trans.data[:, self.tag2int[PAD]] = -10000. # no transition from PAD except to PAD
        self.trans.data[self.tag2int[PAD], :] = -10000. # no transition to PAD except from EOS
        self.trans.data[self.tag2int[PAD], self.tag2int[STOP_TAG]] = 0.
        self.trans.data[self.tag2int[PAD], self.tag2int[PAD]] = 0.

    def forward(self, h, mask): # forward algorithm
        # initialize forward variables in log space
        score = torch.Tensor(h.size(0), self.num_tags).fill_(-10000.).to(self.DEVICE) # [B, C]
        score[:, self.tag2int[START_TAG]] = 0.
        trans = self.trans.unsqueeze(0) # [1, C, C]
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            emit_t = h[:, t].unsqueeze(2) # [B, C, 1]

            #print(score.unsqueeze(1).shape)
            #print (emit_t.shape)
            #print (trans.shape)

            score_t = score.unsqueeze(1) + emit_t + trans # [B, 1, C] -> [B, C, C]
            score_t = log_sum_exp(score_t) # [B, C, C] -> [B, C]
            #print (score_t)
            score = score_t * mask_t + score * (1 - mask_t)
        score = log_sum_exp(score + self.trans[self.tag2int[STOP_TAG]])
        return score # partition function

    def score(self, h, y, mask): # calculate the score of a given sequence
        score = torch.Tensor(h.size(0)).fill_(0.).to(self.DEVICE)

        startTensor = torch.LongTensor([self.tag2int[START_TAG]] * h.size(0)).to(self.DEVICE)

        y = torch.cat([startTensor.unsqueeze(1), y], 1).to(self.DEVICE)

        h = h.unsqueeze(3)
        trans = self.trans.unsqueeze(2)
        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t]
            emit_t = torch.cat([h[t, y[t + 1]] for h, y in zip(h, y)])
            trans_t = torch.cat([trans[y[t + 1], y[t]] for y in y])
            score += (emit_t + trans_t) * mask_t
        last_tag = y.gather(1, mask.sum(1).long().unsqueeze(1)).squeeze(1)
        score += self.trans[self.tag2int[STOP_TAG], last_tag]
        return score

    def decode(self, h, mask): # Viterbi decoding
        # initialize backpointers and viterbi variables in log space
        bptr = torch.LongTensor().to(self.DEVICE)
        score = torch.Tensor(h.size(0), self.num_tags).fill_(-10000.).to(self.DEVICE)
        score[:, self.tag2int[START_TAG]] = 0.

        for t in range(h.size(1)): # recursion through the sequence
            mask_t = mask[:, t].unsqueeze(1)
            score_t = score.unsqueeze(1) + self.trans # [B, 1, C] -> [B, C, C]
            score_t, bptr_t = score_t.max(2) # best previous scores and tags
            score_t += h[:, t] # plus emission scores
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score = score_t * mask_t + score * (1 - mask_t)
        score += self.trans[self.tag2int[STOP_TAG]]
        best_score, best_tag = torch.max(score, 1)

        # back-tracking
        bptr = bptr.tolist()
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(h.size(0)):
            x = best_tag[b] # best tag
            y = int(mask[b].sum().item())
            for bptr_t in reversed(bptr[b][:y]):
                x = bptr_t[x]
                best_path[b].append(x)
            best_path[b].pop()
            best_path[b].reverse()
        return best_path

