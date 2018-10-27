# -*- encoding: utf-8 -*-

import sys
import time


def timeit(msg=''):
    def timeit_(func):
        def newfunc(*args, **kwargs):
           print(msg + '...', end='', file=sys.stderr) 
           sys.stderr.flush()
           start = time.time()
           res = func(*args, **kwargs)
           t = time.time() - start
           print('Done in {:.4f} s'.format(t), file=sys.stderr)
           return res
        return newfunc
    return timeit_


if __name__ == '__main__':
  pass
