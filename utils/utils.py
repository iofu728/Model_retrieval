# -*- coding: utf-8 -*-
# @Description: utils function
# @Author: gunjianpan
# @Date:   2018-11-13 16:14:18
# @Last Modified by:   gunjianpan
# @Last Modified time: 2018-11-13 16:15:05

import time

start = 0


def begin_time():
    global start
    start = time.time()


def end_time():
    print(time.time() - start)
