# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2018-11-18 22:15:38
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-12-02 16:17:07

import codecs
import gensim
import logging
import numpy as np
import pickle
import theano
import warnings
import SMN.sampleConduct

from gensim.models import KeyedVectors
from random import shuffle
from collections import defaultdict
from utils.constant import floatX
from utils.utils import begin_time, end_time, dump_bigger, load_bigger


logger = logging.getLogger('relevance_logger')
warnings.filterwarnings('ignore')


def build_multiturn_data(trainfile, max_len=100, isshuffle=False):
    revs = []
    vocab = defaultdict(float)
    total = 1
    multiturnDatas = pickle.load(open(trainfile, 'rb'))
    for line in multiturnDatas:
        line = line.replace("\n", "")
        parts = line.strip().split('#')
        lable = parts[0]
        message = ""
        words = set()
        for i in range(1, len(parts) - 1, 1):
            message += "_t_"
            message += parts[i]
            words.update(set(parts[i].split()))

        response = parts[-1]

        data = {"y": lable, "m": message, "r": response}
        revs.append(data)
        total += 1
        if not total % 100000:
            print(total)
        words.update(set(response.split()))

        for word in words:
            vocab[word] += 1
    logger.info("processed dataset with %d question-answer pairs " %
                (len(revs)))
    logger.info("vocab size: %d" % (len(vocab)))
    if isshuffle == True:
        shuffle(revs)
    return revs, vocab, max_len


def build_data(trainfile, max_len=20, isshuffle=False):
    revs = []
    vocab = defaultdict(float)
    total = 1
    with codecs.open(trainfile, 'r', 'utf-8') as f:
        for line in f:
            line = line.replace("\n", "")
            parts = line.strip().split("#")

            topic = parts[0]
            topic_r = parts[1]
            lable = parts[2]
            message = parts[-2]
            response = parts[-1]

            data = {"y": lable, "m": message,
                    "r": response, "t": topic, "t2": topic_r}
            revs.append(data)
            total += 1

            words = set(message.split())
            words.update(set(response.split()))
            for word in words:
                vocab[word] += 1
    logger.info("processed dataset with %d question-answer pairs " %
                (len(revs)))
    logger.info("vocab size: %d" % (len(vocab)))
    if isshuffle is True:
        shuffle(revs)
    return revs, vocab, max_len


class WordVecs(object):
    def __init__(self, fname, vocab, binary, gensim):
        if gensim:
            word_vecs = self.load_gensim(fname, vocab)
        self.k = len(list(word_vecs.values())[0])
        self.W, self.word_idx_map = self.get_W(word_vecs, k=self.k)

    def get_W(self, word_vecs, k=200):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        vocab_size = len(word_vecs)
        word_idx_map = dict()
        W = np.zeros(shape=(vocab_size + 1, k))
        W[0] = np.zeros(k)
        i = 1
        for word in word_vecs:
            W[i] = word_vecs[word]
            word_idx_map[word] = i
            i += 1
        return W, word_idx_map

    def load_gensim(self, fname, vocab):

        model = load_bigger(fname)
        weights = [[0.] * model.vector_size]
        word_vecs = {}
        total_inside_new_embed = 0
        miss = 0
        for pair in vocab:
            word = gensim.utils.to_unicode(pair)
            if word in model:
                total_inside_new_embed += 1
                word_vecs[pair] = np.array([w for w in model[word]])
                #weights.append([w for w in model[word]])
            else:
                miss = miss + 1
                word_vecs[pair] = np.array([0.] * model.vector_size)
                #weights.append([0.] * model.vector_size)
        print('transfer', total_inside_new_embed,
              'words from the embedding file, total', len(vocab), 'candidate')
        print('miss word2vec', miss)
        return word_vecs


def createtopicvec():
    max_topicword = 50
    model = Word2Vec.load_word2vec_format("SMN/trainresult")
    topicmatrix = np.zeros(shape=(100, max_topicword, 100),
                           dtype=floatX)
    file = open("SMN/trainpre")
    i = 0
    miss = 0
    for line in file:
        tmp = line.strip().split(' ')
        for j in range(min(len(tmp), max_topicword)):
            if gensim.utils.to_unicode(tmp[j]) in model.vocab:
                topicmatrix[i, j, :] = model[gensim.utils.to_unicode(tmp[j])]
            else:
                miss = miss + 1

        i = i + 1
    print("miss word2vec", miss)
    return topicmatrix


def ParseSingleTurn():
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    revs, vocab, max_len = build_data("SMN/trainpre", isshuffle=True)
    word2vec = WordVecs("SMN/trainresult", vocab, True, True)
    pickle.dump([revs, word2vec, max_len, createtopicvec()],
                open("smn_data.test", 'wb'))
    logger.info("dataset created!")


def ParseMultiTurn(input_file, utterence_file='SMN/data/utterence_11.pkl', embedding_file='SMN/data/embedding_tencent.pkl', pre_file='SMN/data/smn_11.pkl', block_size=20000):
    """
    prepare data
    """
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    version = begin_time()
    sampleConduct = SMN.sampleConduct.SampleConduct()
    sampleConduct.origin_sample_master(input_file, utterence_file, block_size)
    revs, vocab, max_len = build_multiturn_data(
        utterence_file, isshuffle=False)
    word2vec = WordVecs(embedding_file, vocab, True, True)
    dump_bigger([revs, word2vec, max_len], pre_file)
    logger.info("dataset created!")
    end_time(version)


def ParseMultiTurnTest(input_file, utterence_file='SMN/data/utterence_little2.pkl', embedding_file='SMN/data/embedding_tencent.pkl', pre_file='SMN/data/smn_little2.pkl', block_size=20000):
    """
    prepare data
    """
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    version = begin_time()
    sampleConduct = SMN.sampleConduct.SampleConduct()
    sampleConduct.origin_test_master(input_file, utterence_file, block_size)
    revs, vocab, max_len = build_multiturn_data(
        utterence_file, isshuffle=False)
    word2vec = WordVecs(embedding_file, vocab, True, True)
    pickle.dump([revs, word2vec, max_len], open(pre_file, 'wb'))
    logger.info("dataset created!")
    end_time(version)


def embedding_model_test(input_file, utterence_file='SMN/data/utterence_embed1.pkl', embedding_file='SMN/data/embedding_dataset.pkl', pre_file='SMN/data/smn_embedding.pkl', block_size=1000):
    """
    prepare data
    """
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    version = begin_time()
    sampleConduct = SMN.sampleConduct.SampleConduct()
    sampleConduct.origin_test_master(input_file, utterence_file, block_size)
    sampleConduct.embedding_test_master(utterence_file, embedding_file)
    end_time(version)


def dump_word2vec(input_file='../Tencent_AILab_ChineseEmbedding.txt', output_file='SMN/data/embedding_tencent.pkl'):
    """
    load word2vec model
    """
    model = KeyedVectors.load_word2vec_format(input_file, binary=False)
    dump_bigger(model, output_file)


if __name__ == "__main__":
    ParseMultiTurn()
