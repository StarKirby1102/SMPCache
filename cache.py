#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import csv
import os

def cacheName(loperand, roperand, op):
    '''
    Naming conventions, facilitate cache storage and loading
    '''
    return loperand + "_" + op + "_" + roperand

def cacheSave(id, ciphertext, cacheName):
    '''
    This module cache the data slice to local binary file.

    Note: we store the tuple in ciphertext as S16 type.
    '''
    cache_path = "server/P{}/".format(id) + cacheName + ".sdata"
    sess = tf.Session()
    if type(ciphertext) is np.ndarray:
        tabletobesaved = ciphertext.astype("S16")
    else:
        ciphertext = sess.run(ciphertext)
        tabletobesaved = ciphertext.astype("S16")
    bytesbuffer = tabletobesaved.tobytes()
    fd = open(cache_path, "wb")
    fd.write(bytesbuffer)
    fd.close()
    '''
    Need record the shape of cache, in order to reveal the ciphertext according to *.sdata and shape!
    '''

def cacheLoad(id, cacheName, tshape):
    '''
    This module read and reconstruct the ciphertext table from bytesbuffer
    '''
    cache_path = "server/P{}/".format(id) + cacheName + ".sdata"
    fd = open(cache_path, "rb")
    bytesbuffer = fd.read()
    fd.close()
    cipherTable = np.frombuffer(bytesbuffer, dtype = "S16", count = tshape[0]*tshape[1], offset = 0)
    cipherTable.resize((tshape[0], tshape[1]))
    return cipherTable

def ifCached(id, cacheName):
    '''
    This module judge whether the ciphertext has been cached
    '''
    cache_path = "server/P{}/".format(id) + cacheName + ".sdata"
    return os.path.exists(cache_path)

def judgeTypes(tuple):
    '''
    When compute the result from the operand stack, we should judge the type of the operand:

    There are three types:
    (1) Digit: e.g. '4', '50000'        return 1
    (2) Column: e.g. 'ID', 'deposit'            return 2
    (3) Cipher result: e.g. [[b'ul\xcd\x11\x89\x9d\xba\xe1#'][b'\x9fIa\xcbj%8n#']...]       return 3

    '''
    if type(tuple) != str:
        return 3
    else:
        if tuple.isdigit() == True:
            return 1
        else:
            return 2

def cacheRule(id, loperand, roperand, op):
    '''
    This module judge whether the subquery need to be cached:

    If return true && not has been cached -> cache to local
    If return false -> MPC directly, not cache to local
    '''

    if judgeTypes(loperand) == 2 and judgeTypes(roperand) == 2:
        '''
        Rule1. If there is no pure digit in the subquery, it is compared between columns

        e.g. 'loan > 5000' will return false, 'loan > deposit' will return true
        '''
        return True
    
    '''
    Rule2. 
    '''

    return False