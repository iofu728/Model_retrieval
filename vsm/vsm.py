# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2018-11-11 20:27:41
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-11-13 16:16:28

import math
import numpy as np
import pandas as pd
import threading
import time

from utils.utils import begin_time, end_time


class VSM():
    """
    handle write vsm 馃檳
    """

    def __init__(self):
        self.articleMaps = []
        self.articleNum = 0
        self.process = 0
        self.resultArray = []
        self.wordMaps = {}
        self.preData()

    def preData(self):
        """
        data prepare
        """
        begin_time()
        file_d = open('test3', 'r')
        articles = file_d.readlines()
        threadings = []
        self.articleNum = len(articles)
        self.articleMaps = [None for i in range(self.articleNum)]
        self.resultArray = [None for i in range(self.articleNum)]
        for index in range(self.articleNum):
            work = threading.Thread(target=self.preDataBasic, args=(
                articles[index].strip('\n').rstrip(), index,))
            threadings.append(work)
        for work in threadings:
            work.start()
        for work in threadings:
            work.join()
        end_time()

    def preDataBasic(self, article, articleId):
        """
        prepare data basic in Threading
        @param article: article string
        @param articleId: article id
        """
        words = article.split(' ')
        wordMap = {}
        for word in words:
            if word in wordMap:
                wordMap[word] = wordMap[word] + 1
            else:
                wordMap[word] = 1
        for word in wordMap:
            if word in self.wordMaps:
                self.wordMaps[word] = self.wordMaps[word] + 1
            else:
                self.wordMaps[word] = 1
        self.articleMaps[articleId] = wordMap

    def tfidfTest(self, wordMap):
        """
        calculate tdidf value
        td use Augmented Frequency 0.5 + 0.5 * fre/maxFre
        """

        wordlist = [wordMap[i] for i in [*wordMap]]
        maxFrequency = max(wordlist)
        tf = np.array([0.5 + 0.5 * index / maxFrequency for index in wordlist])
        idf = np.array([math.log(self.articleNum / self.wordMaps[word])
                        for word in [*wordMap]])
        tfidf = tf * idf
        return tfidf

    def tfidf(self, wordMap):
        """
        calculate tdidf value
        td use Augmented Frequency 0.5 + 0.5 * fre/maxFre
        """

        wordlist = [wordMap[i] for i in [*wordMap]]
        maxFrequency = max(wordlist)
        tf = np.array([0.5 + 0.5 * index / maxFrequency for index in wordlist])
        idf = np.array([math.log(self.articleNum / (1 + self.wordMaps[word]))
                        for word in [*wordMap]])
        tfidf = tf * idf
        return tfidf / np.linalg.norm(tfidf, ord=2)

    def preSimilarity(self, wordMap, index):
        """
        align map and then calculate one tfidf
        """
        tempMap = {
            index: wordMap[index] if index in wordMap else 0 for index in self.wordMaps}
        preMap = {**wordMap, **tempMap}
        self.resultArray[index] = self.tfidf(preMap)
        self.process += 1
        if not self.process % 100:
            print(self.process)

    def vsmTest(self):
        """
        once to calaulate vsm
        """
        begin_time()
        threadings = []
        for index in range(self.articleNum):
            work = threading.Thread(target=self.preSimilarity, args=(
                self.articleMaps[index], index,))
            threadings.append(work)
        for work in threadings:
            work.start()
        for work in threadings:
            work.join()
        tempMatrix = np.array(self.resultArray)
        result = tempMatrix.dot(tempMatrix.T)
        df = pd.DataFrame(result)
        df.to_csv("vsm1.csv", header=False)
        end_time()

    def preSimilarityTest(self, wordMap1, wordMap2):
        """
        align map and then calculate one tfidf
        """
        tempMap1 = {
            index: wordMap1[index] if index in wordMap1 else 0 for index in wordMap2}
        preMap1 = {**wordMap1, **tempMap1}
        return self.tfidfTest(preMap1)

    def similarity(self, wordMap1, wordMap2, types):
        """
        calculate similarity by cos distance
        @Param types: distance calculate type
                    =0 Cos Distance
                    =1 Chebyshev Distance
                    =2 Manhattan Distance
                    =3 Euclidean Distance
        """
        tfidf1 = self.preSimilarityTest(wordMap1, wordMap2)
        tfidf2 = self.preSimilarityTest(wordMap2, wordMap1)
        if not types:
            return np.dot(tfidf1, tfidf2) / (np.linalg.norm(tfidf1, ord=2) * np.linalg.norm(tfidf2, ord=2))
        elif types == 1:
            return np.abs(tfidf1 - tfidf2).max()
        elif types == 2:
            return np.sum(np.abs(tfidf1 - tfidf2))
        elif types == 3:
            return np.linalg.norm(tfidf1 - tfidf2)
        else:
            return np.shape(np.nonzero(tfidf1 - tfidf2)[0])[0]

    def vsmCalculate(self):
        """
        calculate vsm
        """
        #: todo write block
        begin_time()
        threadings = []
        for index1 in range(self.articleNum):
            work = threading.Thread(target=self.vsmThread, args=(index1,))
            threadings.append(work)
        for work in threadings:
            work.start()
        for work in threadings:
            work.join()
        end_time()

    def vsmThread(self, index1):
        """
        vsm threading
        """
        nowarticle = self.articleMaps[index1]
        tempResult = []
        for index2 in range(index1, self.articleNum):
            tempResult.append(self.vsmPre(
                nowarticle, self.articleMaps[index2]))

        df = pd.DataFrame({index1: tempResult})
        df.to_csv('vsm.csv', mode='a', header=False)

    def vsmPre(self, wordMap1, wordMap2):
        """
        load data to result
        prevent read block
        """

        self.process += 1
        if not self.process % 100:
            print(self.process)
        return self.similarity(wordMap1, wordMap2, 0)
