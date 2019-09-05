#获取词典
def acquireWordDict(inputPathArr, wordDict):
    for inputPath in inputPathArr:
        input = open(inputPath, 'r', encoding='utf-8', errors='ignore')
        for line in input.readlines():
            if len(line) == 0: continue
            word = line.split(' ')[0]
            if word not in wordDict:
                wordDict.append(word)
        input.close()