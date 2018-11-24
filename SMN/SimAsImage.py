# -*- coding: utf-8 -*-
# @Author: gunjianpan
# @Date:   2018-11-18 21:53:33
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-11-24 13:54:37

import theano
import theano.tensor as T

from NN.Classifier import HiddenLayer2
from NN.CNN import LeNetConvPoolLayer
from utils.utils import kmax_pooling, ortho_weight, shared_common, flatten


def batched_dot(input_1, input_2):
    """
    T.batched_dot by input1 and input2.dimshuffle(0, 2, 1)
    """
    return T.batched_dot(input_1, input_2.dimshuffle(0, 2, 1))


class PoolingSim(object):
    def __init__(self, rng, n_in, n_out, W=None, b=None, activation=T.tanh):
        self.W = shared_common(ortho_weight(100), 'W')
        self.activation = activation
        self.hidden_layer = HiddenLayer2(rng, 2 * 5 * n_in, n_out)

        self.params = [self.W] + self.hidden_layer.params

    def __call__(self, input_l, input_r, batch_size, max_l):
        channel_1 = batched_dot(input_l, input_r)
        channel_2 = batched_dot(T.dot(input_l, self.W), input_r)
        input_value = T.stack([channel_1, channel_2], axis=1)
        poolingoutput = kmax_pooling(
            input_value, [batch_size, 2, max_l, max_l], 5)
        mlp_in = T.flatten(poolingoutput, 2)
        return self.hidden_layer(mlp_in)


class PoolingSim3(object):
    def __init__(self, rng, n_in, n_out, W=None, b=None, activation=T.tanh, hidden_size=100):
        self.W = shared_common(ortho_weight(hidden_size), 'W')
        self.activation = activation
        self.hidden_layer = HiddenLayer2(rng, 2 * 5 * n_in, n_out)

        self.params = [self.W] + self.hidden_layer.params

    def __call__(self, origin_l, origin_r, input_l, input_r, batch_size, max_l):
        channel_1 = batched_dot(origin_l, origin_r)
        channel_2 = batched_dot(T.dot(input_l, self.W), input_r)
        input = T.stack([channel_1, channel_2], axis=1)
        poolingoutput = kmax_pooling(input, [batch_size, 2, max_l, max_l], 5)
        mlp_in = T.flatten(poolingoutput, 2)
        return self.hidden_layer(mlp_in)


class PoolingSim2(object):
    def __init__(self, rng, n_in, n_out, tensor_num=3,
                 activation=T.tanh):
        self.tensor_num = tensor_num
        self.W = []
        for i in range(tensor_num):
            self.W.append(shared_common(ortho_weight(100)))
        self.activation = activation
        self.hidden_layer = HiddenLayer2(rng, tensor_num * 5 * n_in, n_out)

        self.params = self.W + self.hidden_layer.params

    def __call__(self, input_l, input_r, batch_size, max_l):
        channels = []
        for i in range(self.tensor_num):
            channels.append(batched_dot(T.dot(input_l, self.W[i]), input_r))

        input = T.stack(channels, axis=1)
        poolingoutput = kmax_pooling(
            input, [batch_size, self.tensor_num, max_l, max_l], 5)
        mlp_in = T.flatten(poolingoutput, 2)
        return self.hidden_layer(mlp_in)


class ConvSim(object):
    def __init__(self, rng, n_in, n_out, W=None, b=None,
                 activation=T.tanh, hidden_size=100):
        self.W = shared_common(ortho_weight(hidden_size))
        self.activation = activation

        self.conv_layer = LeNetConvPoolLayer(rng, filter_shape=(8, 2, 3, 3),
                                             image_shape=(200, 2, 50, 50), poolsize=(3, 3), non_linear='relu')

        self.hidden_layer = HiddenLayer2(rng, 2048, n_out)
        self.params = [self.W, ] + \
            self.conv_layer.params + self.hidden_layer.params

    def Get_M2(self, input_l, input_r):
        return batched_dot(T.dot(input_l, self.W), input_r)

    def __call__(self, origin_l, origin_r, input_l, input_r):
        channel_1 = batched_dot(origin_l, origin_r)
        channel_2 = batched_dot(T.dot(input_l, self.W), input_r)
        input = T.stack([channel_1, channel_2], axis=1)
        mlp_in = T.flatten(self.conv_layer(input), 2)

        return self.hidden_layer(mlp_in)


class ConvSim2(object):
    def __init__(self, rng, n_in, n_out, W=None, b=None,
                 activation=T.tanh, hidden_size=100):
        self.W = shared_common(ortho_weight(hidden_size))
        self.activation = activation

        self.conv_layer = LeNetConvPoolLayer(rng, filter_shape=(8, 1, 3, 3),
                                             image_shape=(200, 1, 50, 50), poolsize=(3, 3), non_linear='relu')

        self.hidden_layer = HiddenLayer2(rng, 2048, n_out)
        self.params = self.conv_layer.params + self.hidden_layer.params

    def __call__(self, origin_l, origin_r):
        channel_1 = batched_dot(origin_l, origin_r)
        input = channel_1.dimshuffle(0, 'x', 1, 2)
        mlp_in = T.flatten(self.conv_layer(input), 2)

        return self.hidden_layer(mlp_in)
