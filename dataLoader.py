from util import acquireWordDict
import torch
from torch.utils import data

#定义标签字典
tagDict = ['<PAD>', 'B-NAME', 'M-NAME', 'E-NAME', 'O', 'B-CONT', 'M-CONT', 
    'E-CONT', 'B-EDU', 'M-EDU', 'E-EDU', 'B-TITLE', 'M-TITLE', 'E-TITLE', 
    'B-ORG', 'M-ORG', 'E-ORG', 'B-RACE', 'E-RACE', 'B-PRO', 'M-PRO', 'E-PRO', 
    'B-LOC', 'M-LOC', 'E-LOC','S-RACE', 'S-NAME', 'M-RACE', 'S-ORG', 'S-CONT', 
    'S-EDU','S-TITLE', 'S-PRO','S-LOC']

int2tag = {index:element for index, element in enumerate(tagDict)}
tag2int = {element:index for index, element in enumerate(tagDict)}

#定义词典
wordDict = ['<PAD>']
acquireWordDict(['./data/dev.char.bmes', './data/test.char.bmes', './data/train.char.bmes'], wordDict)
int2word = {index:element for index, element in enumerate(wordDict)}
word2int = {element:index for index, element in enumerate(wordDict)}


class NERDataset(data.Dataset):
    '''
    : path 语料路径
    '''
    def __init__(self, path, config):
        self.config = config
        f = open(path, 'r', encoding='utf-8', errors='ignore')
        sentenceList, tagList = [], []
        sentence, tag = [], []
        for line in f.readlines():
            #换行
            if len(line.strip()) == 0:
                if len(sentence) != 0 and len(tag) != 0: 
                    if len(sentence) == len(tag):
                        sentenceList.append(sentence); tagList.append(tag)
                sentence, tag = [], []
            else:
                line = line.strip()
                if len(line.split(' ')) < 1: continue
                #针对submit
                if len(line.split(' ')) == 1: 
                    sentence.append(line); tag.append(int2tag[0]); continue
                sentence.append(line.split(' ')[0])
                tag.append(line.split(' ')[1])
        f.close()
        if len(sentence) != 0 and len(tag) != 0: 
            if len(sentence) == len(tag):
                sentenceList.append(sentence); tagList.append(tag)
        self.sentenceList, self.tagList = sentenceList, tagList
    
    def __len__(self):
        return len(self.sentenceList)

    def __getitem__(self, index):
        sentence, tag = self.sentenceList[index], self.tagList[index]

        sentence = [word2int[element] for element in sentence]

        tag = [tag2int[element] for element in tag]

        return sentence, tag

'''
进行填充
'''
def pad(batch):
    f = lambda x:[element[x] for element in batch]
    lenList = [len(element) for element in f(0)]
    maxLen = max(lenList)
    
    f = lambda x, maxLen:[element[x] + [0] * (maxLen - len(element[x]))  for element in batch]

    return torch.LongTensor(f(0, maxLen)), torch.LongTensor(f(1, maxLen)), lenList