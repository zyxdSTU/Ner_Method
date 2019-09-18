import yaml
from optparse import OptionParser
from tqdm import tgrange
from tqdm import tqdm
from dataLoader import NERDataset
from dataLoader import pad
from dataLoader import tagDict
from dataLoader import int2tag
from dataLoader import tag2int
from dataLoader import wordDict
from dataLoader import word2int
from dataLoader import int2word
from torch.utils import data
import torch.nn as nn
import torch.optim as optim
import torch

from bilstm import BiLSTM
from bilstm import bilstmEval
from bilstm import bilstmTrain
from bilstm_crf import BiLSTM_CRF
from bilstm_crf import bilstmCRFEval
from bilstm_crf import bilstmCRFTrain
from transformer_crf import Transformer_CRF
from transformer_crf import transformerCRFTrain
from transformer_crf import transformerCRFEval
from cnn import CNN
from cnn import cnnTrain
from cnn import cnnEval

import sys
from seqeval.metrics import f1_score, accuracy_score, classification_report


def main(config):
    trainDataPath = config['data']['trainDataPath']
    validDataPath = config['data']['validDataPath']
    testDataPath = config['data']['testDataPath']

    modelName = config['modelName']


    batchSize = config['model']['batchSize']
    epochNum = config['model']['epochNum']
    earlyStop = config['model']['earlyStop']
    learningRate = config['model']['learningRate']
    modelSavePath = config['model']['modelSavePath']
    
    #GPU/CPU
    DEVICE = config['DEVICE']

    trianDataset = NERDataset(trainDataPath, config)
    validDataset = NERDataset(validDataPath, config)
    testDataset = NERDataset(testDataPath, config)
    
    trainIter = data.DataLoader(dataset = trianDataset,
                                 batch_size = batchSize,
                                 shuffle = True,
                                 num_workers = 4,
                                 collate_fn = pad)

    validIter = data.DataLoader(dataset = validDataset,
                                 batch_size = batchSize,
                                 shuffle = False,
                                 num_workers = 4,
                                 collate_fn = pad)

    testIter = data.DataLoader(dataset = testDataset,
                                 batch_size = batchSize,
                                 shuffle = False,
                                 num_workers = 4,
                                 collate_fn = pad)

    if modelName == 'bilstm':
        net = BiLSTM(config)
        train = bilstmTrain
        eval = bilstmEval
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    
    if modelName == 'bilstm_crf':
        net = BiLSTM_CRF(config)
        train = bilstmCRFTrain
        eval = bilstmCRFEval
    
    if modelName == 'transformer_crf':
        net = Transformer_CRF(config)
        train = transformerCRFTrain
        eval = transformerCRFEval
    
    if modelName == 'cnn':
        net = CNN(config)
        train = cnnTrain
        eval = cnnEval

    net = net.to(DEVICE)

    lossFunction = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=learningRate,betas=(0.9, 0.999), eps=1e-08)

    earlyNumber, beforeLoss, maxScore = 0, sys.maxsize, -1

    #开始训练
    for epoch in range(epochNum):

        print('第%d次迭代: ' % epoch)

        totalLoss = train(net, trainIter, optimizer=optimizer, criterion=lossFunction, DEVICE=DEVICE)
        print ('训练损失为: %f' % totalLoss)

        totalLoss, f1Score = eval(net,validIter,criterion=lossFunction, DEVICE=DEVICE)

        if f1Score > maxScore:
            maxScore = f1Score
            torch.save(net.state_dict(), modelSavePath)

        print ('验证损失为:%f   f1Score:%f / %f' % (totalLoss, f1Score, maxScore))

        if f1Score < maxScore:
            earlyNumber += 1
            print('earyStop: %d/%d' % (earlyNumber, earlyStop))
        else:
            earlyNumber = 0
        if earlyNumber >= earlyStop: break
        print ('\n')


    #加载最优模型
    net.load_state_dict(torch.load(modelSavePath))
    totalLoss, f1Score = eval(net, testIter, criterion=lossFunction, DEVICE=DEVICE)
    print ('测试损失为: %f, f1Score: %f' %(totalLoss, f1Score))



if __name__ == "__main__":
    #指定model
    optParser = OptionParser()
    optParser.add_option('-m', '--model', action = 'store', type='string', dest ='modelName')
    option, args = optParser.parse_args()
    
    f = open('./config.yml', encoding='utf-8', errors='ignore')
    config = yaml.load(f)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config['DEVICE'] = DEVICE
    config['modelName'] = option.modelName
    f.close()
    main(config)
    
