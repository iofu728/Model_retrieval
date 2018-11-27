# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2018-11-20 16:20:41
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-11-25 12:15:55

import codecs
import numpy as np
import pickle
import random
import threading


from utils.utils import begin_time, end_time, flatten


class SampleConduct(object):
    """
    1. multi to one line
    2. generate negative sample
    """

    def __init__(self):
        self.content = []
        self.response = []
        self.pre = []
        self.origin_sample = []

    def origin_sample_master(self, input_file, output_file, block_size=100000):
        """
        get origin sample master for mult-Theading
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
                target=self.origin_sample_agent, args=(start, end,))
            threadings.append(work)
            start = end + 1
            end = min(num - 1, block_size * (block + 1))
        for work in threadings:
            work.start()
        for work in threadings:
            work.join()
        self.content = list(flatten(self.content))
        self.response = list(flatten(self.response))
        self.pre = list(flatten(self.pre))
        pickle.dump([self.content, self.response, self.pre],
                    open(output_file, "wb"))
        end_time(version)

    def origin_sample_agent(self, start, end):
        """
        origin sample agent for theadings
        """

        temp_context = ''
        last_index = ''
        content = []
        response = []
        pre = []
        for index in range(start, end):
            tempword = self.origin_sample[index]
            if tempword == '\r\n':
                content.append(temp_context)
                response.append(last_index)
                pre.append("1#" + temp_context + last_index + '\n')
                aa = random.randint(0, len(response) - 1)
                pre.append("0#" + temp_context + response[aa] + '\n')
                aaa = random.randint(0, len(response) - 1)
                pre.append("0#" + temp_context + response[aaa] + '\n')
                temp_context = ''
                last_index = ''
            else:
                if len(last_index):
                    temp_context += (last_index + '#')
                last_index = tempword[:-1].strip()
        self.content.append(content)
        self.response.append(response)
        self.pre.append(pre)

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
                    pre.append("1#" + temp_context + last_index + '\n')
                    temp_context = ''
                else:
                    if len(last_index):
                        temp_context += (last_index + '#')
                    last_index = tempword[:-1].strip()
            pickle.dump([content, response, pre], open(output_file, "wb"))
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
