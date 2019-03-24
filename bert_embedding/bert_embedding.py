# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2019-03-23 15:40:19
# @Last Modified by:   gunjianpan
# @Last Modified time: 2019-03-24 19:38:15
import io
import numpy as np
import pandas as pd
import mxnet as mx
import gluonnlp

from mxnet.gluon.data import DataLoader, Dataset
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform
from nltk.corpus import wordnet as wn
from scipy.stats import spearmanr


def to_unicode(s):
    return unicode(s, 'utf-8')


class BertEmbeddingDataset(Dataset):

    def __init__(self, sentences, transform=None):

        self.sentences = sentences
        self.transform = transform

    def __getitem__(self, idx):
        sentence = (self.sentences[idx], 0)
        if self.transform:
            return self.transform(sentence)
        else:
            return sentence

    def __len__(self):
        return len(self.sentences)


class BertEmbedding(object):

    def __init__(self, ctx=mx.cpu(), model='bert_12_768_12',
                 dataset_name='book_corpus_wiki_en_uncased',
                 max_seq_length=25, batch_size=256):

        self.ctx = ctx
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.bert, self.vocab = gluonnlp.model.get_model(model,
                                                         dataset_name=dataset_name,
                                                         pretrained=True, ctx=self.ctx,
                                                         use_pooler=False,
                                                         use_decoder=False,
                                                         use_classifier=False)

    def __call__(self, sentences, oov_way='avg'):
        return self.embedding(sentences, oov_way='avg')

    def embedding(self, sentences, oov_way='avg'):

        data_iter = self.data_loader(sentences=sentences)
        batches = []
        for token_ids, valid_length, token_types in data_iter:
            token_ids = token_ids.as_in_context(self.ctx)
            valid_length = valid_length.as_in_context(self.ctx)
            token_types = token_types.as_in_context(self.ctx)
            sequence_outputs = self.bert(token_ids, token_types,
                                         valid_length.astype('float32'))
            for token_id, sequence_output in zip(token_ids.asnumpy(),
                                                 sequence_outputs.asnumpy()):
                batches.append((token_id, sequence_output))
        return self.oov(batches, oov_way)

    def data_loader(self, sentences, shuffle=False):
        tokenizer = BERTTokenizer(self.vocab)
        transform = BERTSentenceTransform(tokenizer=tokenizer,
                                          max_seq_length=self.max_seq_length,
                                          pair=False)
        dataset = BertEmbeddingDataset(sentences, transform)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle)

    def oov(self, batches, oov_way='avg'):
        sentences = []
        for token_ids, sequence_outputs in batches:
            tokens = []
            tensors = []
            oov_len = 1
            for token_id, sequence_output in zip(token_ids, sequence_outputs):
                if token_id == 1:
                    break
                if token_id in (2, 3):
                    continue
                token = self.vocab.idx_to_token[token_id]
                if token.startswith('##'):
                    token = token[2:]
                    tokens[-1] += token
                    if oov_way == 'last':
                        tensors[-1] = sequence_output
                    else:
                        tensors[-1] += sequence_output
                    if oov_way == 'avg':
                        oov_len += 1
                else:
                    if oov_len > 1:
                        tensors[-1] /= oov_len
                        oov_len = 1
                    tokens.append(token)
                    tensors.append(sequence_output)
            if oov_len > 1:
                tensors[-1] /= oov_len
            sentences.append((tokens, tensors))
        return sentences


def simiarity(s1, s2):
    return np.dot(s1, s2) / (np.linalg.norm(s1, ord=2) * np.linalg.norm(s2, ord=2))


def load_data():
    data = pd.read_csv("word2vec/data/MTURK-771.csv")
    wordsList = np.array(data.iloc[:, [0, 1]])
    simScore = np.array(data.iloc[:, [2]])

    word_set = set([*np.array(data.iloc[:, 0]), *np.array(data.iloc[:, 1])])
    sents = list(word_set)
    for i, (word1, word2) in enumerate(wordsList):
        synsets1 = [jj for ii in wn.synsets(word1) for jj in ii.lemma_names()]
        synsets2 = [jj for ii in wn.synsets(word2) for jj in ii.lemma_names()]
        sents = [*sents, *synsets1, *synsets2]
    sents = list(set(sents))
    word2id = {ww: ii for ii, ww in enumerate(sents)}
    id2word = {ii: ww for ii, ww in enumerate(sents)}

    return wordsList, simScore, sents, word2id, id2word


def evaluation_basic_bert(simScore, wordsList, result, word2id):
    """
    basic bert spearmanr
    """
    predScoreList = np.zeros((len(simScore), 1))
    pm = {ii: res[1][0] for ii, res in enumerate(result)}
    for i, (word1, word2) in enumerate(wordsList):
        predScoreList[i, 0] = simiarity(pm[word2id[word1]], pm[word2id[word2]])
    print(spearmanr(simScore, predScoreList))


def evaluation_bert_wordnet(simScore, wordsList, result, word2id, types):
    """
    bert spearmanr base on wordnet
    """
    predScoreList = np.zeros((len(simScore), 1))
    pm = {ii: res[1][0] for ii, res in enumerate(result)}
    for i, (word1, word2) in enumerate(wordsList):
        # print("process #%d words pair [%s,%s]" % (i, word1, word2))
        if types:
            count = 0
        else:
            temp_score = simiarity(pm[word2id[word1]], pm[word2id[word2]])
        synsets1 = [jj for ii in wn.synsets(word1) for jj in ii.lemma_names()]
        synsets2 = [jj for ii in wn.synsets(word2) for jj in ii.lemma_names()]
        synsets1 = list(set(synsets1))
        synsets2 = list(set(synsets2))
        for synset1 in synsets1:
            for synset2 in synsets2:
                score = simiarity(pm[word2id[synset1]], pm[word2id[synset2]])
                if types:
                    predScoreList[i, 0] += score
                    count += 1
                else:
                    if score > temp_score:
                        temp_score = score
        if types:
            predScoreList[i, 0] = predScoreList[i, 0] / count
        else:
            predScoreList[i, 0] = temp_score
    # print(predScoreList)
    print(spearmanr(simScore, predScoreList))


if __name__ == '__main__':
    np.set_printoptions(threshold=5)
    context = mx.cpu()
    models = ['bert_12_768_12', 'bert_24_1024_16']
    datasets = ['book_corpus_wiki_en_uncased', 'book_corpus_wiki_en_cased',
                'wiki_multilingual_uncased', 'wiki_multilingual_cased', 'wiki_cn_cased']
    max_seq_length = 25
    batch_size = 256
    oov_way = 'avg'
    wordsList, simScore, sents, word2id, id2word = load_data()

    for ii, model in enumerate(models):
        for jj, dataset in enumerate(datasets):
            if ii and jj > 1:
                break
            bert_embedding = BertEmbedding(ctx=context, model=model, dataset_name=dataset,
                                           max_seq_length=max_seq_length, batch_size=batch_size)
            result = bert_embedding(sents, oov_way=oov_way)

            evaluation_basic_bert(simScore, wordsList, result, word2id)
            evaluation_bert_wordnet(simScore, wordsList, result, word2id, 0)
            evaluation_bert_wordnet(simScore, wordsList, result, word2id, 1)
