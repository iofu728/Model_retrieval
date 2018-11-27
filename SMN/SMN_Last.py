# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2018-11-18 22:08:40
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-11-27 11:14:24

import pickle
import numpy as np
import theano
import theano.tensor as T

from NN.Classifier import LogisticRegression
from NN.RNN import GRU
from NN.Optimization import Adam
from SMN.SimAsImage import ConvSim
from utils.constant import floatX, max_turn
from utils.utils import begin_time, end_time, flatten, shared_common, spend_time


def get_idx_from_sent_msg(sents, word_idx_map, max_l=50, mask=False):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    turns = []
    for sent in sents.split('_t_'):
        x = [0] * max_l
        x_mask = [0.] * max_l
        words = sent.split()
        length = len(words)
        for i, word in enumerate(words):
            if max_l - length + i < 0:
                continue
            if word in word_idx_map:
                x[max_l - length + i] = word_idx_map[word]
            x_mask[max_l - length + i] = 1
        if mask:
            x += x_mask
        turns.append(x)

    final = [0.] * (max_l * 2 * max_turn)
    for i in range(max_turn):
        if max_turn - i <= len(turns):
            for j in range(max_l * 2):
                final[i * (max_l * 2) + j] = turns[-(max_turn - i)][j]
    return final


def get_idx_from_sent(sent, word_idx_map, max_l=50, mask=False):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = [0] * max_l
    x_mask = [0.] * max_l
    words = sent.split()
    length = len(words)
    for i, word in enumerate(words):
        if max_l - length + i < 0:
            continue
        if word in word_idx_map:
            x[max_l - length + i] = word_idx_map[word]
        x_mask[max_l - length + i] = 1
    if mask:
        x += x_mask
    return x


def _dropout_from_layer(rng, layer, p):
    """p is the probablity of dropping a unit
    """
    srng = theano.tensor.shared_randomstreams.RandomStreams(
        rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
    mask = srng.binomial(n=1, p=1 - p, size=layer.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
    output = layer * T.cast(mask, floatX)
    return output


def predict(datasets,
            U,  # pre-trained word embeddings
            n_epochs=5, batch_size=20, max_l=100, hidden_size=100, word_embedding_size=100,
            session_hidden_size=50, session_input_size=50, model_name='SMN_last3.bin'):  # for optimization
    """
    return: a list of dicts of lists, each list contains (ansId, groundTruth, prediction) for a question
    """
    hiddensize = hidden_size
    U = U.astype(dtype=floatX)
    rng = np.random.RandomState(3435)
    lsize, rsize = max_l, max_l

    sessionmask = T.matrix()
    lx = []
    lxmask = []
    for i in range(max_turn):
        lx.append(T.matrix())
        lxmask.append(T.matrix())

    index = T.lscalar()
    rx = T.matrix('rx')
    rxmask = T.matrix()
    y = T.ivector('y')
    Words = shared_common(U, "Words")
    llayer0_input = []
    for i in range(max_turn):
        llayer0_input.append(Words[T.cast(lx[i].flatten(), dtype="int32")]
                             .reshape((lx[i].shape[0], lx[i].shape[1], Words.shape[1])))

    rlayer0_input = Words[T.cast(rx.flatten(), dtype="int32")].reshape(
        (rx.shape[0], rx.shape[1], Words.shape[1]))  # input: word embeddings of the mini batch

    dev_set, test_set = datasets[1], datasets[2]

    q_embedding = []
    offset = 2 * lsize

    val_set_lx = []
    val_set_lx_mask = []
    for i in range(max_turn):
        val_set_lx.append(shared_common(np.asarray(
            dev_set[:, offset * i:offset * i + lsize], dtype=floatX)))
        val_set_lx_mask.append(
            shared_common(np.asarray(dev_set[:, offset * i + lsize:offset * i + 2 * lsize], dtype=floatX)))

    val_set_rx = shared_common(np.asarray(
        dev_set[:, offset * max_turn:offset * max_turn + lsize], dtype=floatX))
    val_set_rx_mask = shared_common(
        np.asarray(dev_set[:, offset * max_turn + lsize:offset * max_turn + 2 * lsize], dtype=floatX))
    val_set_session_mask = shared_common(np.asarray(
        dev_set[:, -max_turn - 1:-1], dtype=floatX))
    val_set_y = shared_common(np.asarray(dev_set[:, -1], dtype="int32"))

    val_dic = {}
    for i in range(max_turn):
        val_dic[lx[i]] = val_set_lx[i][index *
                                       batch_size:(index + 1) * batch_size]
        val_dic[lxmask[i]] = val_set_lx_mask[i][index *
                                                batch_size:(index + 1) * batch_size]
    val_dic[rx] = val_set_rx[index * batch_size:(index + 1) * batch_size]
    val_dic[sessionmask] = val_set_session_mask[index *
                                                batch_size:(index + 1) * batch_size]
    val_dic[rxmask] = val_set_rx_mask[index *
                                      batch_size:(index + 1) * batch_size]
    val_dic[y] = val_set_y[index * batch_size:(index + 1) * batch_size]

    sentence2vec = GRU(n_in=word_embedding_size,
                       n_hidden=hiddensize, n_out=hiddensize)

    for i in range(max_turn):
        q_embedding.append(sentence2vec(llayer0_input[i], lxmask[i], True))
    r_embedding = sentence2vec(rlayer0_input, rxmask, True)

    pooling_layer = ConvSim(
        rng, max_l, session_input_size, hidden_size=hiddensize)

    poolingoutput = []
    test = theano.function([index], pooling_layer(llayer0_input[-4], rlayer0_input,
                                                  q_embedding[i], r_embedding), givens=val_dic, on_unused_input='ignore')

    for i in range(max_turn):
        poolingoutput.append(pooling_layer(llayer0_input[i], rlayer0_input,
                                           q_embedding[i], r_embedding))

    session2vec = GRU(n_in=session_input_size,
                      n_hidden=session_hidden_size, n_out=session_hidden_size)
    res = session2vec(T.stack(poolingoutput, 1), sessionmask)
    classifier = LogisticRegression(res, session_hidden_size, 2, rng)

    cost = classifier.negative_log_likelihood(y)
    error = classifier.errors(y)
    opt = Adam()
    params = classifier.params
    params += sentence2vec.params
    params += session2vec.params
    params += pooling_layer.params
    params += [Words]

    load_params(params, model_name)
    print(params)

    predict = classifier.predict_prob

    val_model = theano.function(
        [index], [y, predict, cost, error], givens=val_dic, on_unused_input='ignore')
    f = open('result.txt', 'w')
    loss = 0.
    for minibatch_index in range(int(datasets[1].shape[0] / batch_size)):
        a, b, c, d = val_model(minibatch_index)
        print(c)
        loss += c
        for i in range(batch_size):
            f.write(str(b[i][1]))
            f.write('\n')
            # f.write(str(a[i]))
            # f.write('\n')
            # print(b[i][1], a[i], c[i], d[i])
    print(loss / (datasets[1].shape[0] / batch_size))


def load_params(params, filename):
    f = open(filename, 'rb')
    num_params = pickle.load(f)
    for p, w in zip(params, num_params):
        p.set_value(w.astype(floatX), borrow=True)
    print("load successfully")


def train(datasets,
          U,  # pre-trained word embeddings
          n_epochs=5, batch_size=20, max_l=100, hidden_size=100, word_embedding_size=100,
          session_hidden_size=50, session_input_size=50, model_name='SMN_last3.bin'):
    hiddensize = hidden_size
    U = U.astype(dtype=floatX)
    rng = np.random.RandomState(3435)
    lsize, rsize = max_l, max_l
    sessionmask = T.matrix()
    lx = []
    lxmask = []
    for i in range(max_turn):
        lx.append(T.matrix())
        lxmask.append(T.matrix())

    index = T.lscalar()
    rx = T.matrix('rx')
    rxmask = T.matrix()
    y = T.ivector('y')
    Words = shared_common(U, "Words")
    llayer0_input = []
    for i in range(max_turn):
        llayer0_input.append(Words[T.cast(lx[i].flatten(), dtype="int32")]
                             .reshape((lx[i].shape[0], lx[i].shape[1], Words.shape[1])))

    rlayer0_input = Words[T.cast(rx.flatten(), dtype="int32")].reshape(
        (rx.shape[0], rx.shape[1], Words.shape[1]))  # input: word embeddings of the mini batch

    train_set, dev_set, test_set = datasets[0], datasets[1], datasets[2]

    train_set_lx = []
    train_set_lx_mask = []
    q_embedding = []
    offset = 2 * lsize
    for i in range(max_turn):
        train_set_lx.append(shared_common(np.asarray(
            train_set[:, offset * i:offset * i + lsize], dtype=floatX)))
        train_set_lx_mask.append(shared_common(np.asarray(
            train_set[:, offset * i + lsize:offset * i + 2 * lsize], dtype=floatX)))
    train_set_rx = shared_common(np.asarray(
        train_set[:, offset * max_turn:offset * max_turn + lsize], dtype=floatX))
    train_set_rx_mask = shared_common(np.asarray(
        train_set[:, offset * max_turn + lsize:offset * max_turn + 2 * lsize], dtype=floatX))
    train_set_session_mask = shared_common(np.asarray(
        train_set[:, -max_turn - 1:-1], dtype=floatX))
    train_set_y = shared_common(np.asarray(train_set[:, -1], dtype="int32"))

    val_set_lx = []
    val_set_lx_mask = []
    for i in range(max_turn):
        val_set_lx.append(shared_common(np.asarray(
            dev_set[:, offset * i:offset * i + lsize], dtype=floatX)))
        val_set_lx_mask.append(shared_common(np.asarray(
            dev_set[:, offset * i + lsize:offset * i + 2 * lsize], dtype=floatX)))

    val_set_rx = shared_common(np.asarray(
        dev_set[:, offset * max_turn:offset * max_turn + lsize], dtype=floatX))
    val_set_rx_mask = shared_common(np.asarray(
        dev_set[:, offset * max_turn + lsize:offset * max_turn + 2 * lsize], dtype=floatX))
    val_set_session_mask = shared_common(np.asarray(
        dev_set[:, -max_turn - 1:-1], dtype=floatX))
    val_set_y = shared_common(np.asarray(dev_set[:, -1], dtype="int32"))

    dic = {}
    for i in range(max_turn):
        dic[lx[i]] = train_set_lx[i][index *
                                     batch_size:(index + 1) * batch_size]
        dic[lxmask[i]] = train_set_lx_mask[i][index *
                                              batch_size:(index + 1) * batch_size]
    dic[rx] = train_set_rx[index * batch_size:(index + 1) * batch_size]
    dic[sessionmask] = train_set_session_mask[index *
                                              batch_size:(index + 1) * batch_size]
    dic[rxmask] = train_set_rx_mask[index *
                                    batch_size:(index + 1) * batch_size]
    dic[y] = train_set_y[index * batch_size:(index + 1) * batch_size]

    val_dic = {}
    for i in range(max_turn):
        val_dic[lx[i]] = val_set_lx[i][index *
                                       batch_size:(index + 1) * batch_size]
        val_dic[lxmask[i]] = val_set_lx_mask[i][index *
                                                batch_size:(index + 1) * batch_size]
    val_dic[rx] = val_set_rx[index * batch_size:(index + 1) * batch_size]
    val_dic[sessionmask] = val_set_session_mask[index *
                                                batch_size:(index + 1) * batch_size]
    val_dic[rxmask] = val_set_rx_mask[index *
                                      batch_size:(index + 1) * batch_size]
    val_dic[y] = val_set_y[index * batch_size:(index + 1) * batch_size]

    sentence2vec = GRU(n_in=word_embedding_size,
                       n_hidden=hiddensize, n_out=hiddensize)

    for i in range(max_turn):
        q_embedding.append(sentence2vec(llayer0_input[i], lxmask[i], True))
    r_embedding = sentence2vec(rlayer0_input, rxmask, True)

    pooling_layer = ConvSim(
        rng, max_l, session_input_size, hidden_size=hiddensize)

    poolingoutput = []
    test = theano.function([index], pooling_layer(llayer0_input[-4], rlayer0_input,
                                                  q_embedding[i], r_embedding), givens=val_dic, on_unused_input='ignore')
    # print(test(0).shape)
    # print(test(0))

    for i in range(max_turn):
        poolingoutput.append(pooling_layer(llayer0_input[i], rlayer0_input,
                                           q_embedding[i], r_embedding))

    session2vec = GRU(n_in=session_input_size,
                      n_hidden=session_hidden_size, n_out=session_hidden_size)
    res = session2vec(T.stack(poolingoutput, 1), sessionmask)
    classifier = LogisticRegression(res, session_hidden_size, 2, rng)

    cost = classifier.negative_log_likelihood(y)
    error = classifier.errors(y)
    opt = Adam()
    params = classifier.params
    params += sentence2vec.params
    params += session2vec.params
    params += pooling_layer.params
    params += [Words]
    grad_updates = opt.Adam(cost=cost, params=params,
                            lr=0.001)  # opt.sgd_updates_adadelta(params, cost, lr_decay, 1e-8, sqr_norm_lim)
    # print(index.type())
    train_model = theano.function(
        [index], cost, updates=grad_updates, givens=dic, on_unused_input='ignore')
    val_model = theano.function(
        [index], [cost, error], givens=val_dic, on_unused_input='ignore')
    best_dev = 1.
    n_train_batches = int(datasets[0].shape[0] / batch_size)
    for i in range(n_epochs):
        cost = 0
        total = 0.
        for minibatch_index in np.random.permutation(range(n_train_batches)):
            batch_cost = train_model(minibatch_index)
            total += 1
            cost += batch_cost
            if not total % 50:
                print(total, cost / total)
        cost = cost / n_train_batches
        print("echo %d loss %f" % (i, cost))

        cost = 0
        errors = 0
        j = 0
        for minibatch_index in range(int(datasets[1].shape[0] / batch_size)):
            tcost, terr = val_model(minibatch_index)
            cost += tcost
            errors += terr
            j += 1
        if not j:
            j = 1
        cost /= j
        errors /= j
        if cost < best_dev:
            best_dev = cost
            save_params(params, model_name)
        print("echo %d dev_loss %f" % (i, cost))
        print("echo %d dev_accuracy %f" % (i, 1 - errors))


def save_params(params, filename):
    num_params = [p.get_value() for p in params]
    f = open(filename, 'wb')
    pickle.dump(num_params, f)


def get_session_mask(sents):
    session_mask = [0.] * max_turn
    turns = []
    for sent in sents.split('_t_'):
        words = sent.split()
        if len(words) > 0:
            turns.append(len(words))

    for i in range(max_turn):
        if max_turn - i <= len(turns):
            session_mask[-(max_turn - i)] = 1.
    return session_mask


def make_data_train(revs, word_idx_map, max_l=50, validation_num=50000, block_size=100000):
    """
    Transforms sentences into a 2-d matrix.
    """
    version = begin_time()
    train, val, test, threadings = [], [], [], []
    num = len(revs)
    start = 0
    end = min(block_size, num - 1)
    for block in range(int(num / block_size) + 1):
        work = threading.Thread(target=make_data_theading, args=(
            revs, word_idx_map, max_l, validation_num, start, end,))
        threadings.append(work)
        start = end + 1
        end = min(num - 1, block_size * (block + 1))
    for work in threadings:
        work.start()
    for work in threadings:
        work.join()
        temptrain, tempval = work.get_result()
        train.append(temptrain)
        val.append(tempval)

    train = list(flatten(train))
    val = list(flatten(val))
    train = np.array(train, dtype="int")
    val = np.array(val, dtype="int")
    test = np.array(test, dtype="int")
    print('trainning data', len(train), 'val data',
          len(val), 'spend time:', spend_time(version))
    return [train, val, test]


def make_data_theading(revs, word_idx_map, max_l, validation_num, start, end):
    """
    make data theading
    """
    version = begin_time()
    temptrain, tempval, temptest = [], [], []

    for index in range(start, end):
        rev = revs[index]
        sent = get_idx_from_sent_msg(rev["m"], word_idx_map, max_l, True)
        sent += get_idx_from_sent(rev["r"], word_idx_map, max_l, True)
        sent += get_session_mask(rev["m"])
        sent.append(int(rev["y"]))
        if index > validation_num:
            temptrain.append(sent)
        else:
            tempval.append(sent)
    return temptrain, tempval


def make_data(revs, word_idx_map, max_l=50, filter_h=3, val_test_splits=[2, 3], validation_num=100000):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, val, test = [], [], []
    for rev in revs:
        sent = get_idx_from_sent_msg(rev["m"], word_idx_map, max_l, True)
        sent += get_idx_from_sent(rev["r"], word_idx_map, max_l, True)
        sent += get_session_mask(rev["m"])
        sent.append(int(rev["y"]))
        if len(val) > validation_num:
            train.append(sent)
        else:
            val.append(sent)

    train = np.array(train, dtype="int")
    val = np.array(val, dtype="int")
    test = np.array(test, dtype="int")
    print('trainning data', len(train), 'val data',
          len(val), 'spend time:', spend_time(version))
    return [train, val, test]


if __name__ == "__main__":
    version = begin_time()
    train_flag = False
    max_word_per_utterence = 50
    x = pickle.load(open("smn_data_result.test", "rb"))
    revs, wordvecs, max_l2 = x[0], x[1], x[2]

    datasets = make_data(revs, wordvecs.word_idx_map,
                         max_l=max_word_per_utterence)

    if train_flag == True:
        train(datasets, wordvecs.W, batch_size=200,
              max_l=max_word_per_utterence, hidden_size=200, word_embedding_size=200)
    else:
        predict(datasets, wordvecs.W, batch_size=200,
                max_l=max_word_per_utterence, hidden_size=200, word_embedding_size=200)
    end_time(version)
