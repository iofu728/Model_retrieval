# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-03-24 12:35:43
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-03-24 16:27:10
from nltk.corpus import wordnet as wn

import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def load_data():
    data = pd.read_csv("wordnet/data/MTURK-771.csv")
    wordsList = np.array(data.iloc[:, [0, 1]])
    simScore = np.array(data.iloc[:, [2]])

    return wordsList, simScore


def similarity(word1, word2, types):
    # print("process #%d words pair [%s,%s]" % (i, word1, word2))
    pred_score = 0
    synsets1 = wn.synsets(word1)
    synsets2 = wn.synsets(word2)
    if types:
        count = 0
    else:
        temp_score = 0
    for synset1 in synsets1:
        for synset2 in synsets2:
            score = synset1.path_similarity(synset2)
            if score is not None:
                if types:
                    pred_score += score
                    count += 1
                else:
                    if score > temp_score:
                        temp_score = score

    return (pred_score / count if count else 0) if types else temp_score


def evaluate_basic_wordnet(simScore, wordsList, types):
    predScoreList = np.zeros((len(simScore), 1))

    for i, (word1, word2) in enumerate(wordsList):
        # print("process #%d words pair [%s,%s]" % (i, word1, word2))
        predScoreList[i, 0] = similarity(word1, word2, types)
    types = 'Mean 'if types else 'Max '
    print(types, 'Basic wordNet: ', spearmanr(simScore, predScoreList))


def evaluate_list_wordnet(simScore, wordsList, types, type_sub):
    predScoreList = np.zeros((len(simScore), 1))

    for i, (word1, word2) in enumerate(wordsList):
        # print("process #%d words pair [%s,%s]" % (i, word1, word2))
        waitlist = [similarity(word1, word2, types)]
        synsets1 = [jj for ii in wn.synsets(word1) for jj in ii.lemma_names()]
        synsets2 = [jj for ii in wn.synsets(word2) for jj in ii.lemma_names()]
        synsets1 = list(set(synsets1))
        synsets2 = list(set(synsets2))

        for synset1 in synsets1:
            for synset2 in synsets2:
                waitlist.append(similarity(synset1, synset2, type_sub))
        waitlist = pd.DataFrame(waitlist, columns=['1'])
        if types:
            predScoreList[i, 0] = waitlist['1'].mean()
        else:
            predScoreList[i, 0] = waitlist['1'].max()
    types = 'Mean 'if types else 'Max '
    type_sub = 'Mean 'if type_sub else 'Max '
    print(types, type_sub, 'List wordNet: ',
          spearmanr(simScore, predScoreList))


if __name__ == '__main__':
    wordsList, simScore = load_data()
    evaluate_basic_wordnet(simScore, wordsList, 0)
    evaluate_basic_wordnet(simScore, wordsList, 1)
    evaluate_list_wordnet(simScore, wordsList, 0, 0)
    evaluate_list_wordnet(simScore, wordsList, 0, 1)
    evaluate_list_wordnet(simScore, wordsList, 1, 0)
    evaluate_list_wordnet(simScore, wordsList, 1, 1)
