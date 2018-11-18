# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2018-11-18 16:19:51
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-11-18 19:29:06

import logging
import numpy as np
import pickle
import theano
import theano.tensor as T

from theano.tensor.nnet import conv
from utils.constant import floatX
from utils.utils import shared_common


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


class LeNetConvPoolLayer(object):
    """
    convolutional and pool layer
    """

    def __init__(self, rng, filter_shape, image_shape, poolsize=(2, 2), non_linear="tanh"):
        """
        @param rng: random number generate
        @param filter_shape: (number of filters, num of input feature maps,
                              filter height, filter width)
        @param iamge_shape: (batch size, num of input feature maps,
                             image height, image width)
        @param poolsize: downsampling factor
        """

        print('image shape', image_shape)
        print('filter shape', filter_shape)
        assert image_shape[1] == filter_shape[1]
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear

        fan_in = np.prod(filter_shape[1:])
        fan_out = (filter_shape[0] *
                   np.prod(filter_shape[2:]) / np.prod(poolsize))
        if self.non_linear == "none" or self.non_linear == "relu":
            self.W = shared_common(np.asarray(rng.uniform(
                low=-0.01, high=0.01, size=filter_shape), dtype=floatX), "W_conv")
        else:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = shared_common(np.asarray(rng.uniform(
                low=-W_bound, high=W_bound, size=filter_shape), dtype=floatX), "W_conv")
        b_values = np.zeros((filter_shape[0],), dtype=floatX)
        self.b = shared_common(b_values, "b_conv")
        self.params = [self.W, self.b]

    def __call__(self, input_value):
        """
        activation function
        """
        conv_out = conv.conv2d(input=input_value, filters=self.W,
                               filter_shape=self.filter_shape, image_shape=self.image_shape)
        if self.non_linear == "tanh" or self.non_linear == "relu":
            temp_out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
            conv_out_tanh = T.tanh(
                temp_out) if self.non_linear == "tanh" else ReLU(temp_out)
            self.output = T.signal.pool.pool_2d(
                input=conv_out_tanh, ds=self.poolsize, ignore_border=True, mode="max")
        else:
            pooled_out = T.signal.pool.pool_2d(
                input=conv_out, ds=self.poolsize, ignore_border=True, mode="max")
            self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.output

    def predict(self, new_data, batch_size):
        """
        predict new data
        """
        image_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])
        conv_out = conv.conv2d(input=new_data, filters=self.W,
                               filter_shape=self.filter_shape, image_shape=image_shape)
        if self.non_linear == "tanh" or self.non_linear == "relu":
            temp_out = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
            conv_out_tanh = T.tanh(
                temp_out) if self.non_linear == "tanh" else ReLU(temp_out)
            output = T.signal.pool.pool_2d(
                input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = T.signal.pool.pool_2d(
                input=conv_out, ds=self.poolsize, ignore_border=True)
            output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return output
