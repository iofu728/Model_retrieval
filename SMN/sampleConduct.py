# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2018-11-20 16:20:41
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-12-20 11:17:15

import codecs
import logging
import numpy as np
import pickle
import queue
import random
import threading


from utils.utils import begin_time, end_time, flatten, spend_time, load_bigger, unifom_vector, unique_randomint

logger = logging.getLogger('relevance_logger')


class SampleConduct(object):
    """
    1. multi to one line
    2. generate negative sample
    """

    def __init__(self):
        self.content = {}
        self.response = {}
        self.pre = {}
        self.origin_sample = []
        self.test = []
        self.word2vec = []
        self.wordresult = {}
        self.dev = []
        self.train = []

    def origin_sample_master(self, input_file, output1_file, output2_file, block_size=100000, valnum=10000):
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
        content = [self.content[k] for k in sorted(self.content.keys())]
        self.content = sum(content, [])
        response = [self.response[k] for k in sorted(self.response.keys())]
        self.response = sum(response, [])
        # pre = [self.pre[k] for k in sorted(self.pre.keys())]
        # self.pre = sum(pre, [])
        totalnum = len(self.response)
        for index in range(len(self.content)):
            context = self.content[index]
            if index <= valnum:
                self.dev.append("1#" + context + self.response[index])
            else:
                self.train.append("1#" + context + self.response[index])
            otherindexs = np.random.randint(0, totalnum, 2)
            for otherindex in otherindexs:
                while otherindex == index:
                    otherindex = np.random.randint(0, totalnum, 1)[0]
                if index <= valnum:
                    self.dev.append("0#" + context + self.response[otherindex])
                else:
                    self.train.append(
                        "0#" + context + self.response[otherindex])
        pickle.dump(self.train, open(output1_file, "wb"))
        pickle.dump(self.dev, open(output2_file, "wb"))
        end_time(version)

    def onetime_master(self, input_file, output_file, block_size=900000, test_size=2000):
        """
        by numpy
        """
        version = begin_time()
        with codecs.open(input_file, 'r', 'utf-8') as f:
            self.origin_sample = f.readlines()
        threadings = []
        num = 0
        for index, line in enumerate(self.origin_sample):
            num += 1
        start = 0
        end = min(block_size, num - 1)
        block_num = int(num / block_size) + 1
        print('Thread Begin. ', num)
        for block in range(block_num):
            while self.origin_sample[end] != '\r\n' and end < num - 1:
                end += 1
            work = threading.Thread(
                target=self.origin_sample_agent, args=(start, end, block,))
            threadings.append(work)
            start = end + 1
            end = min(num - 1, block_size * (block + 1))
        print('point 1')
        for work in threadings:
            work.start()
        for work in threadings:
            work.join()
        print('Thread Over.')
        return self.content, self.response
        content = np.hstack(np.array(list(self.content.values())))
        totalnum = len(content)
        print(totalnum)
        randomIndexs = unique_randomint(0, totalnum, test_size)
        otherIndexs = np.setdiff1d(np.arange(totalnum), randomIndexs)
        pre_content = content[otherIndexs]
        test_content = content[randomIndexs]
        del content
        gc.collect()
        response = np.hstack(np.array(list(self.response.values())))
        test_response = [response[index] + '\n' + list2str(
            response[unique_randomint(0, totalnum, 9, [index])]) + '\n' for index in randomIndexs]
        otherIndexs = np.setdiff1d(np.arange(totalnum), randomIndexs)

        pre_response = response[otherIndexs]
        max_dtype = max(pre_content.dtype, pre_response.dtype)
        pre_next = pre_content.astype(
            max_dtype) + pre_response.astype(max_dtype)
        with open(output_file + 'seq_replies.txt', 'wb') as f:
            f.write(list2str(test_response))
        with open(output_file + 'seq_context.txt', 'wb') as f:
            f.write(list2str(test_content))
        with open(output_file + 'train.txt', 'wb') as f:
            f.write(list2str(pre_next))
        end_time(version)

    def twotime_master(self, input_file, output_file, block_size=900000, test_size=2000):
        """
        by not using numpy
        """
        version = begin_time()
        with codecs.open(input_file, 'r', 'utf-8') as f:
            self.origin_sample = f.readlines()
        threadings = []
        num = 0
        for index, line in enumerate(self.origin_sample):
            num += 1
        start = 0
        end = min(block_size, num - 1)
        block_num = int(num / block_size) + 1
        print('Thread Begin. ', num)
        for block in range(block_num):
            while self.origin_sample[end] != '\r\n' and end < num - 1:
                end += 1
            work = threading.Thread(
                target=self.origin_sample_agent, args=(start, end, block,))
            threadings.append(work)
            start = end + 1
            end = min(num - 1, block_size * (block + 1))
        print('point 1')
        for work in threadings:
            work.start()
        for work in threadings:
            work.join()
        print('Thread Over.')
        content = sum(list(self.content.values()), [])
        response = sum(list(self.response.values()), [])
        totalnum = len(content)
        print(totalnum)
        randomIndexs = unique_randomint(0, totalnum, test_size)
        otherIndexs = np.setdiff1d(np.arange(totalnum), randomIndexs)
        pre_next = [content[index] + response[index] for index in otherIndexs]
        test_content = [content[index] for index in randomIndexs]

        test_response = [response[index] + list2str([response[indexs].replace('\n', '') for indexs in unique_randomint(
            0, totalnum, 9, [index])]) + '\n' for index in randomIndexs]

        with open(output_file + 'seq_replies.txt', 'w') as f:
            f.write(list2str(test_response))
        with open(output_file + 'seq_context.txt', 'w') as f:
            f.write(list2str(test_content))
        with open(output_file + 'train.txt', 'w') as f:
            f.write(list2str(pre_next))
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
        pre = []
        num = 0
        for index in range(start, end):
            tempword = self.origin_sample[index]
            if tempword == '\r\n':
                num += 1
                content.append(temp_context)
                response.append(last_index)
                # pre.append(temp_context + last_index)
                temp_context = ''
                last_index = ''
            else:
                if len(last_index):
                    temp_context += last_index
                last_index = tempword[:-1].strip() + '\n'
        self.content[block] = content
        self.response[block] = response
        # self.pre[block] = pre

    def origin_sample_direct(self, input_file, output_file):
        """
        origin sample direct no theading
        """

        version = begin_time()
        with codecs.open(input_file, 'r', 'utf-8') as f:
            temp_context = ''
            last_index = ''
            content = []
            response = []
            pre = []
            for tempword in f:
                if tempword == '\r\n':
                    content.append(temp_context)
                    response.append(last_index)
                    pre.append("1#" + temp_context + last_index)
                    temp_context = ''
                else:
                    if len(last_index):
                        temp_context += (last_index + '#')
                    last_index = tempword[:-1].strip()
            pickle.dump(pre, open(output_file, "wb"))
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
