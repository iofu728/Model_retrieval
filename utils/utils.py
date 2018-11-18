# -*- coding: utf-8 -*-
# @Description: utils function
# @Author: gunjianpan
# @Date:   2018-11-13 16:14:18
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-11-18 18:23:13

import time

start = 0
spendList = []


def begin_time():
    global start
    start = time.time()


def end_time_avage():
    termSpend = time.time() - start
    spendList.append(termSpend)
    print(str(termSpend)[0:5] + ' ' +
          str(sum(spendList) / len(spendList))[0:5])


def end_time():
    termSpend = time.time() - start
    print(str(termSpend)[0:5])


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
