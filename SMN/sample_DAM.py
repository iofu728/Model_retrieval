# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2018-12-23 10:54:27
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-01-15 20:50:46

import codecs
import logging
import numpy as np
import pickle
import queue
import random
import threading

from collections import Counter
from utils.utils import begin_time, end_time, flatten, spend_time, load_bigger, unifom_vector, unique_randomint

logger = logging.getLogger('relevance_logger')

def LCS(long_word):
    """
    deal with very long repeat word
    """
    long_word = long_word.replace('丷','')
    if len(long_word) < 4:
        return long_word
    begin_num =1
    for index in range(1, len(long_word) - 1):
        if long_word[:index] == long_word[index:2*index]:
            return long_word[:index]
    return long_word[:10]

def compute_ngrams(word, min_n, max_n):
    #BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
    extended_word =  word
    ngrams = []
    for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
        for i in range(0, len(extended_word) - ngram_length + 1):
            ngrams.append(extended_word[i:i + ngram_length])
    return list(set(ngrams))

class SampleConduct(object):
    """
    1. multi to one line
    2. generate negative sample
    """

    def __init__(self):
        self.word_map = {}
        self.content = {}
        self.response = {}
        self.pre = {}
        self.origin_sample = []
        self.test = []
        self.word2vec = []
        self.wordresult = {}
        self.dev = []
        self.train = []
        self.word2id = {}
        self.embedding = []

    def word2ids(self, input_file, embedding_file, output1_file='SMN/data/weibo/word2id.pkl', output2_file='SMN/data/weibo/word_embedding.pkl', output3_file='SMN/data/weibo/word2id', min_n = 1, max_n = 3):
        """
        word 2 id
        """
        version = begin_time()
        with codecs.open(input_file, 'r', 'utf-8') as f:
            origin_sample = f.readlines()
        word_embedding = load_bigger(embedding_file)
        words = []
        word_map = {}
        embedding_lists = []

        word_map['_OOV_'] = 0
        word_map['_EOS_'] = 1
        embedding_lists.append([0] * 200)
        embedding_lists.append([0] * 200)
        for index in origin_sample:
            if index == '\r\n':
                continue
            words += [LCS(idx) for idx in index.replace('\r\n', '').split()]
            # words.update(set(index.replace('\r\n', '').split()))
        words = Counter(words)
        words = [index for index in words]
        word2id = ['_OOV_ 0','_EOS_ 1']
        word_size = word_embedding.wv.syn0[0].shape[0]

        print('Step 2: Begin')
        index_num = 2
        for idx, index in enumerate(words):
            if index in word_map:
                continue
            if index in word_embedding.wv.vocab.keys():
                word_map[index] = index_num
                index_num += 1
                word2id.append(index + ' ' + str(word_map[index]))
                embedding_lists.append(word_embedding[index].astype('float32'))
            else:
                ngrams = compute_ngrams(index, min_n = min_n, max_n = max_n)
                word_vec = np.zeros(word_size, dtype=np.float32)
                ngrams_found = 0
                ngrams_single = [ng for ng in ngrams if len(ng) == 1]
                ngrams_more = [ng for ng in ngrams if len(ng) > 1]
                for ngram in ngrams_more:
                    if ngram in word_embedding.wv.vocab.keys():
                        word_vec += word_embedding[ngram]
                        ngrams_found += 1
                if ngrams_found == 0:
                    for ngram in ngrams_single:
                        if ngram in word_embedding.wv.vocab.keys():
                            word_vec += word_embedding[ngram]
                            ngrams_found += 1
                if word_vec.any():
                    word_vec /= max(1, ngrams_found)
                    word_map[index] = index_num
                    index_num += 1
                    word2id.append(index + ' ' + str(word_map[index]))
                    embedding_lists.append(word_vec)
        print(index_num)
        with open(output3_file, 'w') as f:
            f.write(list2str(word2id))
        print('Step 2: Over')

        # return embedding_lists, word_map
        pickle.dump(embedding_lists, open(output2_file, "wb"))
        pickle.dump(word_map, open(output1_file, "wb"))
        end_time(version)


    def origin_sample_master(self, input_file, output_file='SMN/data/weibo/train_data_small.pkl', word2id_file='SMN/data/weibo/word2id.pkl', embedding_file='SMN/data/weibo/word_embedding.pkl', block_size=900000, small_size=100000):
        """
        the master of mult-Theading for get origin sample
        """
        version = begin_time()
        with codecs.open(input_file, 'r', 'utf-8') as f:
            self.origin_sample = f.readlines()
        self.word2id = pickle.load(open(word2id_file, 'rb'))
        # self.embedding = pickle.load(open(embedding_file, 'rb'))

        threadings = []
        num = len(self.origin_sample)
        start = 0
        end = min(block_size, num - 1)
        for block in range(int(num / block_size) + 1):
            while self.origin_sample[end] != '\r\n' and end < num - 1:
                end += 1
            work = threading.Thread(
                target=self.origin_sample_agent, args=(start, end, block,))
            threadings.append(work)
            start = end + 1
            end = min(num - 1, block_size * (block + 1))
        for work in threadings:
            work.start()
        for work in threadings:
            work.join()
        content_order = self.content.keys()
        content_order.sort()
        response_order = self.response.keys()
        response_order.sort()

        content = sum(list(self.content.values()), [])
        response = sum(list(self.response.values()), [])

        totalnum = len(content)
        print(totalnum)
        # return totalnum
        randomIndexs = unique_randomint(0, totalnum, small_size)
        y = [1, 0, 0] * small_size
        c= []
        r = []
        for index in randomIndexs:
            c.append(content[index])
            c.append(content[index])
            c.append(content[index])
            r.append(response[index])
            r.append(response[unique_randomint(0, totalnum, 1, [index])[0]])
            r.append(response[unique_randomint(0, totalnum, 1, [index])[0]])

        train_data = {'y': y, 'r': r, 'c': c}
        pickle.dump(train_data, open(output_file, "wb"))
        end_time(version)

    def origin_test_master(self, input_file, output_file, block_size=100000, test_size=2000):
        """
        the master of mult-Theading for get origin sample
        """
        version = begin_time()
        with codecs.open(input_file, 'r', 'utf-8') as f:
            self.origin_sample = f.readlines()
        threadings = []
        num = len(self.origin_sample)
        start = 0
        end = min(block_size, num - 1)
        for block in range(int(num / block_size) + 1):
            while self.origin_sample[end] != '\r\n' and end < num - 1:
                end += 1
            work = threading.Thread(
                target=self.origin_sample_agent, args=(start, end, block, ))
            threadings.append(work)
            start = end + 1
            end = min(num - 1, block_size * (block + 1))
        for work in threadings:
            work.start()
        for work in threadings:
            work.join()
        content = [self.content[k] for k in sorted(self.content.keys())]
        self.content = sum(content, [])
        response = [self.response[k] for k in sorted(self.response.keys())]
        self.response = sum(response, [])
        totalnum = len(self.content)
        randomlists = np.random.randint(0, totalnum, test_size)
        for index in randomlists:
            temp_context = self.content[index]
            self.test.append("1#" + temp_context + self.response[index])
            otherindexs = np.random.randint(0, totalnum, 9)
            for otherindex in otherindexs:
                while otherindex == index:
                    otherindex = np.random.randint(0, totalnum, 1)[0]
                self.test.append("0#" + temp_context +
                                 self.response[otherindex])
        pickle.dump(self.test, open(output_file, 'wb'))
        end_time(version)

    def origin_sample_agent(self, start, end, block):
        """
        the agent of mult-Theading for get origin sample
        """

        temp_context = []
        last_index = []
        content = []
        response = []
        num = 0
        for index in range(start, end):
            tempword = self.origin_sample[index]
            if tempword == '\r\n':
                num += 1
                content.append(temp_context[:-1])
                # content.append(temp_context[:-1])
                # content.append(temp_context[:-1])
                response.append(last_index[:-1])
                temp_context = []
                last_index = []
            else:
                if len(last_index):
                    temp_context += last_index
                last_index = tempword[:-1].strip().split()
                last_index = [(self.word2id[LCS(index)] if LCS(index) in self.word2id else 0) for index in last_index]
                last_index.append(1)
        # r = []
        # totalnum = len(response)
        # for idx, index in enumerate(response):
        #     r.append(index)
        #     r.append(response[unique_randomint(0, totalnum, 1, [idx])[0]])
        #     r.append(response[unique_randomint(0, totalnum, 1, [idx])[0]])
        self.content[block] = content
        self.response[block] = response

    def origin_t_direct(self, input_file='SMN/data/test_SMN.pkl', output_file='SMN/data/weibo/val_Dataset.pkl', small_size=10000, word2id_file='SMN/data/weibo/word2id.pkl'):
        """
        origin sample direct no theading
        """

        version = begin_time()
        test = pickle.load(open(input_file,'rb'))
        self.word2id = pickle.load(open(word2id_file, 'rb'))
        c = []
        r = []
        num = 0
        for tempword in test:
            words = tempword[2:].split('#')
            contexts = words[:-1]
            replys = words[-1]
            context = []
            reply = []
            for idx,index in enumerate(words):
                if idx:
                    context.append(1)
                temp_context = index.split()
                for temp in temp_context:
                    context.append(self.word2id[LCS(temp)] if LCS(temp) in self.word2id else 0)
            for index in replys:
                temp_reply = index.split()
                for temp in temp_reply:
                    reply.append(self.word2id[LCS(temp)] if LCS(temp) in self.word2id else 0)
            r.append(reply)
            c.append(context)
        y=[1,0,0,0,0,0,0,0,0,0]*(len(test) //10)
        val_data = {'y': y, 'r': r, 'c': c}
        pickle.dump(val_data, open(output_file, "wb"))
        end_time(version)

    def origin_sample_direct(self, input1_file, input2_file, output_file, small_size=10000, word2id_file='SMN/data/weibo/word2id.pkl'):
        """
        origin sample direct no theading
        """

        version = begin_time()
        with codecs.open(input1_file, 'r', 'utf-8') as f:
            sample1 = f.readlines()
        with codecs.open(input2_file, 'r', 'utf-8') as f:
            sample2 = f.readlines()
        self.word2id = pickle.load(open(word2id_file, 'rb'))
        temp_context = []
        last_index = []
        c = []
        r = []
        num = 0
        for tempword in sample1:
            if tempword == '\r\n':
                num += 1
                for idx in range(10):
                    c.append(temp_context + last_index[:-1])
                temp_context = []
                last_index = []
            else:
                if len(last_index):
                    temp_context += last_index
                last_index = tempword[:-1].replace('\"','').replace('\\','').strip().split()
                if '\"' in last_index:
                    print(last_index)
                last_index = [(self.word2id[LCS(index)] if LCS(index) in self.word2id else 0) for index in last_index]
                last_index.append(1)
        # for idx in range(10):
        #     c.append(temp_context + last_index[:-1])
        num = 0
        print(len(sample2))
        for index,tempword in enumerate(sample2):
            if tempword != '\r\n':
                num +=1
                last_index = tempword[:-1].replace('\"','').replace('\\','').strip().split()
                last_index = [(self.word2id[LCS(index)] if LCS(index) in self.word2id else 0) for index in last_index]
                r.append(last_index)
            else:
                if num != 10:
                    r.append(last_index)
                    print(num,index)
                num = 0
        y=[1,0,0,0,0,0,0,0,0,0]*small_size
        val_data = {'y': y, 'r': r, 'c': c}
        pickle.dump(val_data, open(output_file, "wb"))

        end_time(version)

    def origin_result_direct(self, input_file1, input_file2, output_file):
        """
        origin sample direct no theading
        """

        version = begin_time()
        pre = []
        dataset = []
        with codecs.open(input_file1, 'r', 'utf-8') as f:
            temp_context = ''
            last_index = ''
            for tempword in f:
                if tempword == '\r\n':
                    pre.append("1#" + temp_context + last_index)
                    temp_context = ''
                    last_index = ''
                else:
                    if len(last_index):
                        temp_context += (last_index + '#')
                    last_index = tempword[:-1].strip()
        with codecs.open(input_file2, 'r', 'utf-8') as f:
            temp_context = []
            index = 0
            totalnum = len(pre)
            for tempword in f:
                if tempword == '\r\n':
                    if len(temp_context) < 9:
                        continue
                    elif len(temp_context) == 9:
                        if index < totalnum:
                            dataset.append(pre[index] + '#' + temp_context[0])
                        index += 1
                        temp_context = []
                    else:
                        index += 1
                        temp_context = []
                else:
                    temp_context.append(tempword[:-1].strip())
                    if index < totalnum:
                        dataset.append(pre[index] + '#' +
                                       tempword[:-1].replace(u'\ufeff', '').strip())
            pickle.dump([pre, dataset], open(output_file, "wb"))
        end_time(version)

    def calculate_result(self, input_file, output_file, block_size=10):
        """
        calculate result
        """
        version = begin_time()
        with codecs.open(input_file, 'r', 'utf-8') as f:
            with codecs.open(output_file, 'w') as outf:
                results = f.readlines()
                for index in range(int(len(results) / block_size)):
                    pre = results[index * block_size:(index + 1) * block_size]
                    temp_index = np.array(pre).argmax()
                    outf.write(str(temp_index) + '\n')
        end_time(version)

    def calculate_test(self, input_file, block_size=10):
        """
        calculate result
        """
        version = begin_time()
        with codecs.open(input_file, 'r', 'utf-8') as f:
            results = f.readlines()
            totalnum = int(len(results))
            correctnum = 0
            top3num = 0
            for index in range(int(totalnum / block_size)):
                pre = results[index * block_size:(index + 1) * block_size]
                temp_index = np.array(pre).argmax()
                top3 = np.array(pre).argsort()[-3:][::-1]
                if not temp_index:
                    correctnum += 1
                if 0 in top3:
                    top3num += 1
            print(correctnum, top3num, int(totalnum / block_size), spend_time(version), str(
                correctnum / int(totalnum / block_size))[:5], str(top3num / int(totalnum / block_size))[:5])
            return str(correctnum / int(totalnum / block_size))[:5]

    def embedding_test_master(self, input_file, embedding_file, block_size=10000):
        """
        the master of mult-Theading for test by embedding model
        """
        version = begin_time()
        self.word2vec = load_bigger(embedding_file)
        self.origin_sample = load_bigger(input_file)
        threadings = queue.Queue()
        waitthreadings = queue.Queue()
        num = len(self.origin_sample)
        start = 0
        end = min(block_size, num - 1)
        for block in range(int(num / block_size) + 1):
            work = threading.Thread(
                target=self.embedding_test_agent, args=(start, end, block,))
            threadings.put(work)
            start = end + 1
            end = min(num - 1, block_size * (block + 2))
        while not threadings.empty():
            tempwork = threadings.get()
            tempwork.start()
            waitthreadings.put(tempwork)
        while not waitthreadings.empty():
            waitthreadings.get().join()

        result = [self.wordresult[k] for k in sorted(self.wordresult.keys())]
        results = sum(result, [])
        totalnum = int(len(results))
        correctnum = 0
        top3num = 0
        block_sizes = 10
        for index in range(int(totalnum / block_sizes)):
            pre = results[index * block_sizes:(index + 1) * block_sizes]
            temp_index = np.array(pre).argmax()
            top3 = np.array(pre).argsort()[-3:][::-1]
            if not temp_index:
                correctnum += 1
            if 0 in top3:
                top3num += 1
        print(correctnum, top3num, int(totalnum / block_sizes), spend_time(version), str(
            correctnum / int(totalnum / block_sizes))[:5], str(top3num / int(totalnum / block_sizes))[:5])
        end_time(version)

    def embedding_test_agent(self, start, end, block):
        """
        the agent of mult-Theading for test by embedding model
        """
        result = []
        origin_sample = self.origin_sample
        word2vec = self.word2vec
        for index in range(start, end):
            tempword = origin_sample[index].replace("\n", "")
            parts = tempword.strip().split('#')
            context = np.zeros(200)
            reply = np.zeros(200)
            for i in range(1, len(parts) - 1, 1):
                words = parts[i].split()
                for word in words:
                    if word in word2vec:
                        context += word2vec[word]
            for word in parts[-1].split():
                if word in word2vec:
                    reply += word2vec[word]

            result.append(np.dot(
                context, reply) / (np.linalg.norm(context, ord=2) * np.linalg.norm(reply, ord=2)))
        self.wordresult[block] = result


def papp(index):
    tempword = origin_sample[index]
    parts = tempword.strip().split('#')
    context = np.zeros(200)
    reply = np.zeros(200)
    for i in range(1, len(parts) - 1, 1):
        words = parts[i].split()
        for word in words:
            if word in word2vec:
                context += word2vec[word]
    for word in parts[-1].split():
        if word in word2vec:
            reply += word2vec[word]
    return np.dot(context, reply) / (np.linalg.norm(context, ord=2) * np.linalg.norm(reply, ord=2))


class GetWords(object):
    """
    word2vec agent
    """

    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        with open(self.dirname, 'r') as f:
            wordLists = f.readlines()
            for index in wordLists:
                yield index.split()


def list2str(lists):
    """
    list to str
    """
    return str(list(lists)).replace('\'', '').replace('\\n', '\n').replace(', ', '\n')[1:-1]


def preWord2vec(input_file, output_file):
    """
    word bag construction
    """
    version = begin_time()
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = GetWords(input_file)
    model = Word2Vec(sentences, workers=100, min_count=5, size=200)
    model.save(output_file)
    end_time(version)
