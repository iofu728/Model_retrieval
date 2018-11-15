# -*- coding: utf-8 -*-
# @Description: utils function
# @Author: gunjianpan
# @Date:   2018-11-13 16:14:18
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-11-15 13:44:16

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
