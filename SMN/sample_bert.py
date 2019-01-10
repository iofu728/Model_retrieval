# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2018-12-23 10:54:27
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-12-30 16:31:34

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
    long_word = long_word.replace('丷', '')
    if len(long_word) < 4:
        return long_word
    begin_num = 1
    for index in range(1, len(long_word) - 1):
        if long_word[:index] == long_word[index:2 * index]:
            return long_word[:index]
    return long_word[:10]


class SampleConduct(object):
    """
    1. multi to one line
    2. generate negative sample
    """

    def __init__(self):
        self.word_map = {}
        self.content = {}
        self.response = {}
        self.r = {}
        self.pre = {}
        self.origin_sample = []
        self.test = []
        self.word2vec = []
        self.wordresult = {}
        self.dev = []
        self.train = []
        self.word2id = {}
        self.embedding = []

    def word2ids(self, input_file, embedding_file, output1_file='SMN/data/weibo/word2id.pkl', output2_file='SMN/data/weibo/word_embedding.pkl', output3_file='SMN/data/weibo/word2id'):
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
        words = [index for index in words if words[index] > 2]
        word2id = ['_OOV_ 0', '_EOS_ 1']

        print('Step 2: Begin')
        index_num = 2
        for idx, index in enumerate(words):
            if index in word_embedding:
                if index not in word_map:
                    word_map[index] = index_num
                    index_num += 1
                    word2id.append(index + ' ' + str(word_map[index]))
                    embedding_lists.append(
                        list(word_embedding[index].astype('float16')))
            # elif index[:3] in word_embedding:
            #     if index[:3] not in word_map:
            #         word_map[index[:3]] = index_num
            #         word_map[index] = index_num
            #         index_num += 1
            #         word2id.append(index[:3] + ' ' + str(word_map[index[:3]]))
            #         word2id.append(index + ' ' + str(word_map[index]))
            #         embedding_lists.append(list(word_embedding[index[:3]].astype('float16')))
            #     else:
            #         word_map[index] = word_map[index[:3]]
            #         word2id.append(index + ' ' + str(word_map[index]))
            # elif index[:2] in word_embedding:
            #     if index[:2] not in word_map:
            #         word_map[index[:2]] = index_num
            #         word_map[index] = index_num
            #         index_num += 1
            #         word2id.append(index[:2] + ' ' + str(word_map[index[:2]]))
            #         word2id.append(index + ' ' + str(word_map[index]))
            #         embedding_lists.append(list(word_embedding[index[:2]].astype('float16')))
            #     else:
            #         word_map[index] = word_map[index[:2]]
            #         word2id.append(index + ' ' + str(word_map[index]))
            # elif index[:1] in word_embedding:
            #     if index[:1] not in word_map:
            #         word_map[index[:1]] = index_num
            #         word_map[index] = index_num
            #         index_num += 1
            #         word2id.append(index[:1] + ' ' + str(word_map[index[:1]]))
            #         word2id.append(index + ' ' + str(word_map[index]))
            #         embedding_lists.append(list(word_embedding[index[:1]].astype('float16')))
            #     else:
            #         word_map[index] = word_map[index[:1]]
            #         word2id.append(index + ' ' + str(word_map[index]))
        print(index_num)
        with open(output3_file, 'w') as f:
            f.write(list2str(word2id))
        print('Step 2: Over')

        # return embedding_lists, word_map
        pickle.dump(embedding_lists, open(output2_file, "wb"))
        pickle.dump(word_map, open(output1_file, "wb"))
        end_time(version)

    def origin_sample_master(self, input_file, output_file='SMN/data/bert/train.pkl', block_size=900000, small_size=200000):
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
                target=self.origin_sample_agent, args=(start, end, block,))
            threadings.append(work)
            start = end + 1
            end = min(num - 1, block_size * (block + 1))
        for work in threadings:
            work.start()
        for work in threadings:
            work.join()
        response = sum(list(self.response.values()), [])
        content = sum(list(self.content.values()), [])
        totalnum = len(response)
        print(totalnum)
        randomIndexs = unique_randomint(0, totalnum, small_size)
        # otherIndexs = unique_randomint(
        #     0, totalnum, small_size * 2, randomIndexs)
        r = []
        for index in randomIndexs:
            r.append('1#' + content[index] + '#' + response[index])
            r.append('0#' + content[index] + '#' +response[unique_randomint(0, totalnum, 1, [index])[0]])
            r.append('0#' + content[index] + '#' +response[unique_randomint(0, totalnum, 1, [index])[0]])

        pickle.dump(r, open(output_file, "wb"))
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

        temp_context = ''
        last_index = ''
        content = []
        response = []
        num = 0
        for index in range(start, end):
            tempword = self.origin_sample[index]
            if tempword == '\r\n':
                num += 1
                content.append(temp_context[:-5])
                response.append(last_index[:-5])
                temp_context = ''
                last_index = ''
            else:
                if len(last_index):
                    temp_context += last_index
                last_index = tempword[:-1].strip() + '[SEP]'
        # r = []
        # totalnum = len(response)
        # for idx, index in enumerate(content):
        #     r.append('1 ' + index + '#' + response[idx])
        #     r.append('0 ' + index + '#' +
        #              response[unique_randomint(0, totalnum, 1, [idx])[0]])
        #     r.append('0 ' + index + '#' +
        #              response[unique_randomint(0, totalnum, 1, [idx])[0]])
        self.response[block] = response
        self.content[block] = content

    def origin_sample_direct(self, input1_file, input2_file, output_file, small_size=2000):
        """
        origin sample direct no theading
        """

        version = begin_time()
        with codecs.open(input1_file, 'r', 'utf-8') as f:
            sample1 = f.readlines()
        with codecs.open(input2_file, 'r', 'utf-8') as f:
            sample2 = f.readlines()

        temp_context = ''
        last_index = ''
        content = []
        r = []
        for tempword in sample1:
            if tempword == '\n':
                content.append(temp_context + last_index[:-5])
                temp_context = ''
                last_index = ''
            else:
                if len(last_index):
                    temp_context += last_index
                last_index = tempword[:-1].strip() + '[SEP]'
        num = 0
        print(len(sample2))
        for index, tempword in enumerate(sample2):
            if tempword != '\n':
                last_index = tempword[:-1].replace('\"', '').replace('\\', '')
                r.append('0#' + content[num] + '#' + last_index)
            else:
                num += 1
        pickle.dump(r, open(output_file, "wb"))

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
