#!/usr/bin/env python3

import multiprocessing
import sys
import time
import numpy as np
import tensorflow as tf
import csv
import os
from parameters import *

def obviousAssign(cond, x, y):
    '''
    This module will assign the secret share to a variable

    Return cond * x + y - cond * y

    (1) If cond is 0-share, will return y-share
    (2) If cond is 1-share, will return x-share 
    '''
    import latticex.rosetta as rtt

    return rtt.SecureAdd(rtt.SecureMul(cond, rtt.SecureSub(x, y)), y)

def secureMax(ciphertable, attr):
    '''
    This module will return the maximum value of ciphertable[:,attr] in secret share form
    '''
    import latticex.rosetta as rtt

    tableLen = len(ciphertable)

    if tableLen == 1:
        return ciphertable[0][attr]
    
    ciphercolumn = ciphertable[:,attr]
    # print(ciphercolumn)
    maxSS = np.array([ciphercolumn[0]])

    sess = tf.Session()
    for i in range(1, tableLen):
        '''
        Traversing the array and extracts the maximum value obviously
        '''
        b = rtt.SecureGreater(np.array([ciphercolumn[i]]), maxSS)
        maxSS = obviousAssign(b, np.array([ciphercolumn[i]]), maxSS)
        tf.reset_default_graph()
        sess.run(maxSS)
    
    return maxSS


def secureMin(ciphertable, attr):
    '''
    This module will return the minimum value of ciphertable[:,attr] in secret share form
    '''
    import latticex.rosetta as rtt

    tableLen = len(ciphertable)

    if tableLen == 1:
        return ciphertable[0][attr]
    
    ciphercolumn = ciphertable[:,attr]
    minSS = np.array([ciphercolumn[0]])

    sess = tf.Session()
    for i in range(1, tableLen):
        '''
        Traversing the array and extracts the minimum value obviously
        '''
        b = rtt.SecureLess(np.array([ciphercolumn[i]]), minSS)
        minSS = obviousAssign(b, np.array([ciphercolumn[i]]), minSS)
        tf.reset_default_graph()
        sess.run(minSS)
    
    return minSS

def secureSum(ciphertable, attr):
    '''
    This module will return the sum of the target records

    e.g. 'SELECT SUM(AGE) FROM TABLE' will return the sum of AGE
    '''
    import latticex.rosetta as rtt

    tableLen = len(ciphertable)

    if tableLen == 1:
        return ciphertable[0][attr]

    ciphercolumn = ciphertable[:,attr]

    sumSS = np.array(ciphercolumn[0])

    sess = tf.Session()
    for i in range(1, tableLen):
        sumSS = rtt.SecureAdd(sumSS, np.array(ciphercolumn[i]))
        sess.run(sumSS)
        # sumSS.append(rtt.SecureAdd(sumSS[i - 1], np.array(ciphercolumn[i])))
        tf.reset_default_graph()

    
    return sumSS

def secureAVG(id, ciphertable, attr):
    '''
    This module will return the sum of the target records

    e.g. 'SELECT AVG(loan) FROM TABLE' will return the average value of loan
    '''
    import latticex.rosetta as rtt

    tableLen = len(ciphertable)

    sumSS = secureSum(ciphertable, attr)

    if id == 0:
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X([[tableLen]])
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
    elif id == 1:
        rd = np.random.RandomState(1789)
        plaintext = rd.randint(0, 1, (1, 1))
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(plaintext)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
    else:
        rd = np.random.RandomState(1999)
        plaintext = rd.randint(0, 1, (1, 1))
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(plaintext)

    avgSS = rtt.SecureDivide(sumSS, rtx)
    tf.reset_default_graph()

    return avgSS

def test(id):
    sys.argv.extend(["--node_id", "P{}".format(id)])
    import latticex.rosetta as rtt
    rtt.activate("SecureNN")

    rtt.backend_log_to_stdout(False)

    test_table = 'users/user3/S_user3_table.csv'

    TIME_START = time.time()
    param = parameters()

    attr = 5

    if id == 0:
        plaintext = np.loadtxt(open(test_table, encoding = 'utf-8'), str, delimiter = ",", skiprows = 1)[0:param.dataScale]
        '''
        NULL value handling
        '''
        plaintext[np.where(plaintext == '')] = '0'

        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(plaintext)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
    elif id == 1:
        rd = np.random.RandomState(1789)
        plaintext = rd.randint(0, 1, (1, 1))
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(plaintext)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
    else:
        rd = np.random.RandomState(1999)
        plaintext = rd.randint(0, 1, (1, 1))
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(plaintext)
    
    session = tf.Session()

    AGG_START = time.time()
    savg = rtt.SecureMax(rtx[:,attr], 0)
    AGG_END = time.time()
    print('Successfully compute the max')

    print('MPC time:', AGG_END - AGG_START, 's')

    AGG_START = time.time()
    smin = rtt.SecureMin(rtx[:,attr], 0)
    AGG_END = time.time()
    print('Successfully compute the min')

    print('MPC time:', AGG_END - AGG_START, 's')

    AGG_START = time.time()
    savg = rtt.SecureMean(rtx[:,attr], 0)
    AGG_END = time.time()
    print('Successfully compute the sum')

    print('MPC time:', AGG_END - AGG_START, 's')


    AGG_START = time.time()
    savg = rtt.SecureMean(rtx[:,attr], 0)
    AGG_END = time.time()
    print('Successfully compute the avg')

    print('MPC time:', AGG_END - AGG_START, 's')

    TIME_END = time.time()
    print('Successfully test all the secure aggregate, total time:', TIME_END - TIME_START)

    # plaintext = session.run(rtt.SecureReveal(savg))
    # print('avg:', plaintext)
    


p0 = multiprocessing.Process(target = test, args = (0,))
p1 = multiprocessing.Process(target = test, args = (1,))
p2 = multiprocessing.Process(target = test, args = (2,))

p0.daemon = True
p0.start()
p1.daemon = True
p1.start()
p2.daemon = True
p2.start()

p0.join()
p1.join()
p2.join()