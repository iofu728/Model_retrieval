# -*- coding: utf-8 -*-
# @Description: utils function
# @Author: gunjianpan
# @Date:   2018-11-13 16:14:18
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-11-29 16:01:20

import collections
import numpy as np
import pickle
import os
import time
import theano
import theano.tensor as T

from utils.constant import float32, floatX

theano.config.floatX = 'float32'

start = []
spendList = []


def begin_time():
    """
    multi-version time manage
    """
    global start
    start.append(time.time())
    return len(start) - 1


def end_time_avage(version):
    termSpend = time.time() - start[version]
    spendList.append(termSpend)
    print(str(termSpend)[0:5] + ' ' +
          str(sum(spendList) / len(spendList))[0:5])


def end_time(version):
    termSpend = time.time() - start[version]
    print(str(termSpend)[0:5])


def spend_time(version):
    return str(time.time() - start[version])[0:5]


def empty():
    spendList = []


def shared_zeros(shape_value, name=None):
    """
    theano.shared np.zeros
    """
    if name is None:
        return theano.shared(value=np.zeros(shape_value, dtype=floatX), borrow=True)
    return theano.shared(value=np.zeros(shape_value, dtype=floatX), borrow=True, name=name)


def shared_ones(shape_value, name=None):
    """
    theano.shared np.ones
    """
    if name is None:
        return theano.shared(value=np.ones(shape_value, dtype=floatX), borrow=True)
    return theano.shared(value=np.ones(shape_value, dtype=floatX), borrow=True, name=name)


def shared_common(variable, name=None):
    """
    theano.shared common
    """
    if name is None:
        return theano.shared(value=variable, borrow=True)
    return theano.shared(value=variable, borrow=True, name=name)


def scan_common(step, sequences, outputs_info):
    """
    theano scan common
    """
    return theano.scan(step, sequences=sequences, outputs_info=outputs_info)


def scan_dimshuffle(step, shape_x, shape_y, outputs_info):
    """
    theano scan dimshuffle
    """
    return theano.scan(step, sequences=[shape_x.dimshuffle(1, 0, 2), T.addbroadcast(shape_y.dimshuffle(1, 0, 'x'), -1)], outputs_info=outputs_info)


def ReLU(input_value):
    """
    ReLU function
    """
    return T.maximum(0.0, input_value)


def kmax_pooling(input_value, input_shape, k):
    """
    k-max pooling
    """
    sorted_value = T.argsort(input_value, axis=3)
    topmax_indexs = sorted_value[:, :, :, -k:]
    topmax_indexs_sorted = T.sort(topmax_indexs)

    dim0 = T.arange(0, input_shape[0]).repeat(
        input_shape[1] * input_shape[2] * k)
    dim1 = T.arange(0, input_shape[1]).repeat(
        input_shape[2] * k).reshape((1, -1)).repeat(input_shape[0], axis=0).flatten()
    dim2 = T.arange(0, input_shape[2]).repeat(k).reshape(
        (1, -1)).repeat(input_shape[0] * input_shape[1], axis=0).flatten()
    dim3 = topmax_indexs_sorted.flatten()

    return input_value[dim0, dim1, dim2].reshape((input_shape[0], input_shape[1], input_shape[2], k))


def ortho_weight(ndim):
    """
    ortho weight matrix
    """
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(float32)


def norm_weight(nin, nout=None, scale=0.01, ortho=False):
    """
    normalization weight matrix
    """
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype(float32)


def unifom_weight(size, scale=0.1):
    """
    uniform distribution weight matrix
    """
    return np.random.uniform(size=size, low=-scale, high=scale).astype(floatX)


def unifom_vector(ppap):
    """
    unifom verctor
    """
    return np.array(ppap) / np.linalg.norm(ppap, ord=2)


def gloroat_uniform(size):
    fan_in, fan_out = size
    s = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(size=size, low=-s, high=s).astype(floatX)


def flatten(lst):
    """
    multilevel flatten generate
    """
    for item in lst:
        if isinstance(item, collections.Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item


def dump_bigger(data, output_file):
    """
    pickle.dump big file which size more than 4GB
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(data, protocol=4)
    with open(output_file, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx + max_bytes])


def load_bigger(input_file):
    """
    pickle.load big file which size more than 4GB
    """
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(input_file)
    with open(input_file, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)
