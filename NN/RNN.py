# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2018-11-18 10:04:13
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-11-18 16:16:23

import pickle
import datetime
import logging
import numpy as np
import os
import theano
import theano.tensor as T
import time

from sklearn.base import BaseEstimator
from utils.constant import float32, floatX
from utils.utils import begin_time, end_time, end_time_avage


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


def gloroat_uniform(size):
    fan_in, fan_out = size
    s = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(size=size, low=-s, high=s).astype(floatX)


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
        return theano.shared(variable, borrow=True)
    return theano.shared(variable, borrow=True, name=name)


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


class RNN(object):
    """
    Naïve RNN
    """

    def __init__(self, input_l, input_r, n_in, n_hidden, n_out, activation=T.tanh, output_type='real', batch_size=200, input_lm=None, input_rm=None):
        if input_lm is None:
            input_lm = shared_ones((batch_size, 20))
        if input_rm is None:
            input_rm = shared_ones((batch_size, 20))
        self.activation = activation
        self.output_type = output_type

        self.W = shared_common(ortho_weight(n_hidden), 'W')
        self.W_in = shared_common(gloroat_uniform(n_in, n_hidden), 'W_in')

        self.h0 = shared_zeros((batch_size, n_hidden), 'h0')
        self.bh = shared_zeros((batch_size, n_hidden), 'bh')

        self.params = [self.W, self.W_in, self.bh]

        def step(x_t, mask, h_tm1):
            h_tm1 = mask * h_tm1
            h_t = T.tanh(T.dot(x_t, self.W_in) +
                         T.dot(h_tm1, self.W) + self.bh)
            return h_t
        self.h_l, _ = scan_dimshuffle(
            step, input_l, input_lm, shared_zeros((batch_size, n_hidden)))
        self.h_r, _ = scan_dimshuffle(
            step, input_r, input_rm, shared_zeros((batch_size, n_hidden)))
        self.h_l = self.h_l.dimshuffle(1, 0, 2)
        self.h_r = self.h_r.dimshuffle(1, 0, 2)


class LSTM(object):
    """
    LSTM
    """

    def __init__(self, n_in, n_hidden, n_out, activation=T.tanh, inner_activation=T.nnet.sigmoid, output_type='real', batch_size=200):
        self.activation = activation
        self.inner_activation = inner_activation
        self.output_type = output_type

        self.batch_size = batch_size
        self.n_hidden = n_hidden

        self.W_i = shared_common(gloroat_uniform((n_in, n_hidden)))
        self.U_i = shared_common(ortho_weight(n_hidden))
        self.b_i = shared_zeros((n_hidden,))

        self.W_f = shared_common(gloroat_uniform((n_in, n_hidden)))
        self.U_f = shared_common(ortho_weight(n_hidden))
        self.b_f = shared_zeros((n_hidden,))

        self.W_c = shared_common(gloroat_uniform((n_in, n_hidden)))
        self.U_c = shared_common(ortho_weight(n_hidden))
        self.b_c = shared_zeros((n_hidden,))

        self.W_o = shared_common(gloroat_uniform((n_in, n_hidden)))
        self.U_o = shared_common(ortho_weight(n_hidden))
        self.b_o = shared_zeros((n_hidden,))

        self.params = [self.W_i, self.U_i, self.b_i,
                       self.W_c, self.U_c, self.b_c,
                       self.W_f, self.U_f, self.b_f,
                       self.W_o, self.U_o, self.b_o]

    def __call__(self, input_value, input_lm=None, return_list=False):
        """
        activation funcation
        """
        if input_lm is None:
            self.h_l, _ = scan_common(self.step, [input_value.dimshuffle(1, 0, 2), T.ones_like(input_value.dimshuffle(
                1, 0, 2))], [shared_zeros(self.batch_size, self.n_hidden), shared_zeros(self.batch_size, self.n_hidden)])
        else:
            self.h_l, _ = scan_dimshuffle(self.step, input_value, input_lm, [shared_zeros(
                self.batch_size, self.n_hidden), shared_zeros(self.batch_size, self.n_hidden)])
        self.h_l = self.h_l[0].dimshuffle(1, 0, 2)
        if return_list is True:
            return self.h_l
        return self.h_l[:, -1, :]

    def step(self, x_t, mask, h_tm1, c_tm1):

        x_i = T.dot(x_t, self.W_i) + self.b_i
        x_f = T.dot(x_t, self.W_f) + self.b_f
        x_c = T.dot(x_t, self.W_c) + self.b_c
        x_o = T.dot(x_t, self.W_o) + self.b_o

        i = self.inner_activation(x_i + T.dot(h_tm1, self.U_i))
        f = self.inner_activation(x_f + T.dot(h_tm1, self.U_f))
        c = self.activation(x_c + T.dot(h_tm1, self.U_c)) * i + f * c_tm1
        o = self.inner_activation(x_o + T.dot(h_tm1, self.U_o))
        h = o * self.activation(c)

        h = mask * h + (1 - mask) * h_tm1
        c = mask * c + (1 - mask) * c_tm1

        return [h, c]


class GRU(object):
    """
    GRU
    """

    def __init__(self, n_in, n_hidden, n_out, activation=T.tanh, inner_activation=T.nnet.sigmoid, output_type='real', batch_size=200):
        self.activation = activation
        self.inner_activation = inner_activation
        self.output_type = output_type
        self.batch_size = batch_size
        self.n_hidden = n_hidden

        self.W_z = shared_common(gloroat_uniform((n_in, n_hidden)))
        self.U_z = shared_common(ortho_weight(n_hidden))
        self.b_z = shared_zeros((n_hidden,))

        self.W_r = shared_common(gloroat_uniform((n_in, n_hidden)))
        self.U_r = shared_common(ortho_weight(n_hidden))
        self.b_r = shared_zeros((n_hidden,))

        self.W_h = shared_common(gloroat_uniform((n_in, n_hidden)))
        self.U_h = shared_common(ortho_weight(n_hidden))
        self.b_h = shared_zeros((n_hidden,))

        self.params = [self.W_z, self.U_z, self.b_z,
                       self.W_r, self.U_r, self.b_r,
                       self.W_h, self.U_h, self.b_h]

    def __call__(self, input_value, input_lm=None, return_list=False, init_input=None, check_gate=False):
        """
        activation function
        """
        if init_input is None:
            init = shared_zeros((self.batch_size, self.n_hidden))
        else:
            init = init_input

        if check_gate:
            self.h_l, _ = scan_dimshuffle(self.step, input_value, input_lm, [
                                          init, shared_zeros((self.batch_size, self.n_hidden))])
            return [self.h_l[0][:, -1, :], self.h_l[1]]

        if input_lm is None:
            self.h_l, _ = scan_common(
                self.step, [input_value.dimshuffle(1, 0, 2), T.ones_like(input_value.dimshuffle(1, 0, 2))], init)
        else:
            self.h_l, _ = scan_dimshuffle(
                self.step, input_value, input_lm, init)
        self.h_l = self.h_l.dimshuffle(1, 0, 2)
        if return_list is True:
            return self.h_l
        return self.h_l[:, -1, :]

    def step(self, x_t, mask, h_tm1, gate_tm1=None):
        x_z = T.dot(x_t, self.W_z) + self.b_z
        x_r = T.dot(x_t, self.W_r) + self.b_r
        x_h = T.dot(x_t, self.W_h) + self.b_h
        z = self.inner_activation(x_z + T.dot(h_tm1, self.U_z))
        r = self.inner_activation(x_r + T.dot(h_tm1, self.U_r))

        hh = self.activation(x_h + T.dot(r * h_tm1, self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        h = mask * h + (1 - mask) * h_tm1

        if gate_tm1 is None:
            return h
        return [h, r]


class BiGRU(object):
    """
    BiGRU
    """

    def __init__(self, n_in, n_hidden, n_out, activation=T.tanh, inner_activation=T.nnet.sigmoid, output_type='real', batch_size=200):
        self.gru_1 = GRU(n_in, n_hidden, n_out, batch_size=batch_size)
        self.gru_2 = GRU(n_in, n_hidden, n_out, batch_size=batch_size)

        self.params = self.gru_1.params
        self.params += self.gru_2.params

    def __call__(self, input_value, input_lm=None, return_list=False):
        reverse_input = input_value[:, ::-1, :]
        reverse_mask = input_lm[:, ::-1]

        res1 = self.gru_1(input_value, input_lm, return_list)
        if return_list is True:
            res2 = self.gru_2(reverse_input, reverse_mask,
                              return_list)[:, ::-1, :]
            return T.concatenate([res1, res2], 2)
        else:
            res2 = self.gru_2(reverse_input, reverse_mask, return_list)
        return T.concatenate([res1, res2], 1)


if __name__ == "__main__":
    begin_time()
    input_value = T.tensor3()
    input2 = T.matrix()
    rnn = GRU(100, 100, 100, batch_size=47)
    res = rnn(input_value, input2, check_gate=True)
    output_value = theano.function([input_value, input2], [res[1]])
    print(output_value(np.random.rand(47, 20, 100).astype('float32'),
                       np.ones((47, 20)).astype('float32'))[0].shape)
    end_time()
