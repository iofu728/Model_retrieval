# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-03-22 20:34:02
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-03-24 20:46:04

import gensim
import math
import numpy as np
import pandas as pd
import random
import re
import string
import time
import tensorflow as tf

from collections import Counter
from numba import jit
from nltk.corpus import wordnet as wn
from utils.utils import begin_time, end_time
from scipy.stats import spearmanr

data_path = 'word2vec/data/'
checkpoint_path = 'word2vec/checkpoints/'


def preprocess(text, floor=10):
    """
    pre process text
    """
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()

    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > floor]

    return trimmed_words


class SG(object):
    """
    skip gram
    """

    def __init__(self):
        self.origin_file_name = data_path + 'wikipedia.txt'
        self.embedding_size = 300
        self.n_sample = 100
        self.epochs = 10
        self.window_size = 10
        self.batch_size = 1000
        self.n_embedding = 300
        self.load_data()

    def load_data(self, floor=10, upper_ptr=1e-5):
        version = begin_time()

        with open(self.origin_file_name, 'r') as f:
            origin_text = f.read()

        text_list = preprocess(origin_text, floor)

        word_counts = Counter(text_list)
        vocab_list = sorted(word_counts, key=word_counts.get, reverse=True)

        word2id = {word: ii for ii, word in enumerate(vocab_list)}
        id2word = {ii: word for ii, word in enumerate(vocab_list)}

        id_list = [word2id[word] for word in text_list]
        word_counts = Counter(id_list)
        count_len = len(id_list)

        p_drop = {word: (1 - np.sqrt(upper_ptr * count_len / count))
                  for word, count in word_counts.items()}

        train_list = [ww for ww in id_list if p_drop[ww] < np.random.random()]

        print("Total words: {}".format(len(train_list)))
        print("Unique words: {}".format(len(set(train_list))))

        self.word2id = word2id
        self.id2word = id2word
        self.train_list = train_list
        end_time(version)

    def get_target(self, words, idx, window_size=5):
        """
        Get a list of words in a window around an index
        """
        R = np.random.randint(1, window_size + 1)
        start = idx - R if (idx - R) > 0 else 0
        stop = idx + R
        target_words = set(words[start:idx] + words[idx + 1:stop + 1])

        return list(target_words)

    def get_batches(self, words, batch_size, window_size=5):
        """
        Create a generator of word batches
        """

        n_batches = len(words) // batch_size

        words = words[:n_batches * batch_size]

        for idx in range(0, len(words), batch_size):
            x, y = [], []
            batch = words[idx:idx + batch_size]
            for ii in range(len(batch)):
                batch_x = batch[ii]
                batch_y = self.get_target(batch, ii, window_size)
                y.extend(batch_y)
                x.extend([batch_x] * len(batch_y))
            yield x, y

    def module_build(self):
        """
        skip gram
        """
        train_graph = tf.Graph()
        n_vocab = len(self.id2word)
        n_embedding = self.n_embedding
        n_sampled = self.n_sample
        valid_size = 16
        valid_window = 100

        with train_graph.as_default():
            input_vec = tf.placeholder(tf.int32, [None], name='inputs')
            label_vec = tf.placeholder(tf.int32, [None, None], name='labels')

            embedding = tf.Variable(tf.random_uniform(
                [n_vocab, n_embedding], -1, 1))
            embedding_answer = tf.nn.embedding_lookup(embedding, input_vec)

            softmax_w = tf.Variable(tf.truncated_normal(
                [n_vocab, n_embedding], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(n_vocab))

            loss = tf.nn.sampled_softmax_loss(
                softmax_w, softmax_b, label_vec, embedding_answer, n_sampled, n_vocab)

            cost = tf.reduce_mean(loss)
            optimizer = tf.train.AdamOptimizer().minimize(cost)

            valid_sample = np.array(random.sample(
                range(valid_window), valid_size // 2))
            valid_sample = np.append(valid_sample,
                                     random.sample(range(1000, 1000 + valid_window), valid_size // 2))

            valid_dataset = tf.constant(valid_sample, dtype=tf.int32)

            norm = tf.sqrt(tf.reduce_sum(
                tf.square(embedding), 1, keepdims=True))
            normalized_embedding = embedding / norm
            valid_embedding = tf.nn.embedding_lookup(
                normalized_embedding, valid_dataset)
            similarity = tf.matmul(
                valid_embedding, tf.transpose(normalized_embedding))

        return input_vec, label_vec, cost, loss, optimizer, similarity, valid_sample, normalized_embedding, train_graph

    def moudle_train(self):

        input_vec, label_vec, cost, loss, optimizer, similarity, valid_sample, normalized_embedding, train_graph = self.module_build()

        epochs = self.epochs
        batch_size = self.batch_size
        window_size = self.window_size
        train_list = self.train_list

        with train_graph.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph=train_graph) as sess:
            count, loss = [1, 0]
            sess.run(tf.global_variables_initializer())

            for ii in range(1, epochs + 1):
                batches = self.get_batches(train_list, 1000, 10)
                start = time.time()
                for x, y in batches:
                    feed = {input_vec: x, label_vec: np.array(y)[:, None]}
                    train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)

                    loss += train_loss

                    if not count % 300:
                        end = time.time()
                        data = [
                            "Epoch {}/{}".format(ii, epochs),
                            "Iteration: {}".format(count),
                            "Avg. Training loss: {:.4f}".format(loss / 100),
                            "{:.4f} sec/batch".format((end - start) / 100)]
                        with open('log/word2vec.log', 'a') as f:
                            f.write(",".join(data) + '\n')
                        print(",".join(data))
                        loss = 0
                        start = time.time()

                    if not count % 2000:
                        sim = similarity.eval()
                        for i in range(len(valid_sample)):
                            valid_word = self.id2word[valid_sample[i]]
                            top_k = 8
                            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                            log = 'Nearest to %s:' % valid_word
                            for k in range(top_k):
                                close_word = self.id2word[nearest[k]]
                                log = '%s %s,' % (log, close_word)
                            print(log)

                    count += 1

            embed_mat = sess.run(normalized_embedding)
            tf.add_to_collection('embed_mat', embed_mat)
            save_path = saver.save(
                sess, checkpoint_path + "word2vec%d.ckpt" % count, global_step=count)
            self.embed_mat = embed_mat
            print('================')
            print(type(embed_mat))
            print('================')

    def reload_model(self):
        with tf.Session() as sess:
            new_saver = tf.train.import_meta_graph(
                tf.train.latest_checkpoint(checkpoint_path) + '.meta')
            new_saver.restore(
                sess, tf.train.latest_checkpoint(checkpoint_path))
            self.embed_mat = tf.get_collection('embed_mat')[0]

    def load_test_data(self):
        data = pd.read_csv("word2vec/data/MTURK-771.csv")
        wordsList = np.array(data.iloc[:, [0, 1]])
        simScore = np.array(data.iloc[:, [2]])

        word_set = set([*np.array(data.iloc[:, 0]), *np.array(data.iloc[:, 1])])
        sents = list(word_set)
        for i, (word1, word2) in enumerate(wordsList):
            synsets1 = [jj for ii in wn.synsets(
                word1) for jj in ii.lemma_names()]
            synsets2 = [jj for ii in wn.synsets(
                word2) for jj in ii.lemma_names()]
            sents = [*sents, *synsets1, *synsets2]
        sents = list(set(sents))
        word2id = {ww: ii for ii, ww in enumerate(sents)}
        id2word = {ii: ww for ii, ww in enumerate(sents)}

        return wordsList, simScore, sents, word2id, id2word

    def evaluation_basic(self, simScore, wordsList, word2id, mold, model):
        """
        basic word2vec spearmanr
        """
        predScoreList = np.zeros((len(simScore), 1))
        if mold == 2:
            model = self.embed_mat

        for i, (word1, word2) in enumerate(wordsList):
            if mold == 2:
                word1 = self.word2id[word1] if word1 in self.word2id else 0
                word2 = self.word2id[word2] if word2 in self.word2id else 0
            predScoreList[i, 0] = self.simiarity(model[word1], model[word2])
        mold = "Google News Basic: " if mold == 3 else "Word2Vec Basic: "
        print(mold, spearmanr(simScore, predScoreList))

    def simiarity(self, s1, s2):
        return np.dot(s1, s2) / (np.linalg.norm(s1, ord=2) * np.linalg.norm(s2, ord=2))

    def evaluation_wordnet(self, simScore, wordsList, word2id, mold, model, types):
        """
        bert spearmanr base on wordnet
        """
        predScoreList = np.zeros((len(simScore), 1))
        if mold == 2:
            model = self.embed_mat
        for i, (word1, word2) in enumerate(wordsList):
            # print("process #%d words pair [%s,%s]" % (i, word1, word2))
            synsets1 = [jj for ii in wn.synsets(
                word1) for jj in ii.lemma_names()]
            synsets2 = [jj for ii in wn.synsets(
                word2) for jj in ii.lemma_names()]
            synsets1 = list(set(synsets1))
            synsets2 = list(set(synsets2))
            if mold == 2:
                word1 = self.word2id[word1] if word1 in self.word2id else 0
                word2 = self.word2id[word2] if word2 in self.word2id else 0
            waitlist = []

            if word1 in model and word2 in model:
                waitlist.append(self.simiarity(model[word1], model[word2]))

            for synset1 in synsets1:
                for synset2 in synsets2:
                    if mold == 2:
                        synset1 = self.word2id[synset1] if synset1 in self.word2id else 0
                        synset2 = self.word2id[synset2] if synset2 in self.word2id else 0
                    if synset1 in model and synset2 in model:
                        waitlist.append(self.simiarity(
                            model[synset1], model[synset2]))
            waitlist = pd.DataFrame(waitlist, columns=['1'])
            if not len(waitlist):
                predScoreList[i, 0] = 0
            elif types == 2:
                predScoreList[i, 0] = waitlist['1'].median()
            elif types == 1:
                predScoreList[i, 0] = waitlist['1'].mean()
            else:
                predScoreList[i, 0] = waitlist['1'].max()
        # print(predScoreList)
        types = ('Mean ' if types == 1 else 'Median ')if types else 'Max '
        mold = "Google News & WordNet: " if mold == 3 else "Word2Vec & WordNet: "
        print(types, mold, spearmanr(simScore, predScoreList))


if __name__ == '__main__':
    version = begin_time()
    np.random.seed(123)
    mold = 3
    model = None
    sg = SG()
    if mold == 1:
        sg.moudle_train()
        mold = 2
    elif mold == 2:
        sg.reload_model()
    elif mold == 3:
        model = gensim.models.KeyedVectors.load_word2vec_format(
            'word2vec/data/GoogleNews-vectors-negative300.bin', binary=True)

    wordsList, simScore, sents, word2id, id2word = sg.load_test_data()
    sg.evaluation_basic(simScore, wordsList, word2id, mold, model)
    sg.evaluation_wordnet(simScore, wordsList, word2id, mold, model, 0)
    sg.evaluation_wordnet(simScore, wordsList, word2id, mold, model, 1)
    sg.evaluation_wordnet(simScore, wordsList, word2id, mold, model, 2)
    end_time(version)
