#!/usr/bin/env python3

import csv
import os
from pydoc import plain
import sys
import multiprocessing
from venv import logger
import numpy as np
import tensorflow as tf
import time
import random
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from queryParse import parseSQL
from saveCipherTable import *
from loadCipherTable import *
from AST import *
from obliviousSort import *
from secureGroupBy import *
from cache import *
from parameters import *
# from executeSecurePlan import *

'''
In this py file, we test the SQL demo:

SELECT user_id
FROM passwords|P1U...Upasswords|Pm
    GROUP BY CONCAT(user_id, password)
    HAVING COUNT(*)>1

Note: this test just for this SQL, deployer can parse your own SQL to use our MPC-Cache idea
'''

def test_demo2(id):
    sys.argv.extend(["--node_id", "P{}".format(id)])
    import latticex.rosetta as rtt
    rtt.backend_log_to_stdout(False)
    rtt.activate("SecureNN")

    SQL2_START = time.time()
    '''
    Step1. group by the ciphertable according to attr, return the sorted ciphertable and compare_result
    '''

    if id == 0:
        '''
        Generate the 1's SS
        '''
        plaintext = [[0], [1]]
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(plaintext)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
    elif id == 1:
        test_table = 'users/user0/S_user0_test.csv'
        plaintext = np.loadtxt(open(test_table), delimiter = ",", skiprows = 1)
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(plaintext)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
    else:
        test_table = 'users/user0/S_user0_test.csv'
        plaintext = np.loadtxt(open(test_table), delimiter = ",", skiprows = 1)
        plaintext_concat = []
        for i in range(len(plaintext)):
            plaintext_concat.append([plaintext[i][0]*pow(10, plaintext[i][1]//10 + 1) + plaintext[i][1]])
        plaintext_concat = np.array(plaintext_concat)
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(plaintext_concat)
    
    print('Successfully load the data')

    MPC_START = time.time()
    sorted_rtxdata, compare_result = secureGroupBy(rtz, 0)
    rty = oblivious_odd_even_merge_sort(0, rty, 0)
    compare_result = np.append(compare_result, rtx[1])

    print('Successfully compute the oblivious sort and private compare')

    sess = tf.Session()
    tensor_0 = tf.convert_to_tensor(rtx[0])
    tensor_1 = tf.convert_to_tensor(rtx[1])
    '''
    Step2. compute the SS compare array D_0 = sorted_data * (1 - compare_result)
    '''
    D_0 = rtt.SecureLogicalXor(tensor_1, compare_result)
    D_0 = rtt.SecureMul(D_0, rty[:,0])
    D_0 = sess.run(D_0)

    '''
    Step3. compute the COUNT(*) for D_1

    C_ = rtt.SecureEqual(D_0, 0)

    counter += 1

    D_1 = (1 - C_) * counter

    update the counter: counter = C_ * counter
    '''
    counter = tf.convert_to_tensor(rtx[0])

    C_ = rtt.SecureEqual(D_0, tensor_0)
    C_ = sess.run(C_)

    D_1 = []
    for i in range(len(rty)):
        counter = rtt.SecureAdd(counter, tensor_1)
        D_1.append(sess.run(rtt.SecureMul(rtt.SecureLogicalXor(C_[i], tensor_1), counter)))
        counter = rtt.SecureMul(counter, C_[i])
        counter = sess.run(counter)
    D_1 = np.array(D_1)

    SQL2_END = time.time()
    print('MPC total time:', SQL2_END - MPC_START, 's')
    print('SQL2\'s total time:', SQL2_END - SQL2_START, 's')

    # print(sess.run(rtt.SecureReveal(sorted_rtxdata)))
    # print(sess.run(rtt.SecureReveal(compare_result)))
    # print(sess.run(rtt.SecureReveal(D_0)))
    # print(sess.run(rtt.SecureReveal(C_)))
    # print(sess.run(rtt.SecureReveal(D_1)))

p0 = multiprocessing.Process(target = test_demo2, args = (0,))
p1 = multiprocessing.Process(target = test_demo2, args = (1,))
p2 = multiprocessing.Process(target = test_demo2, args = (2,))

p0.daemon = True
p0.start()
p1.daemon = True
p1.start()
p2.daemon = True
p2.start()

p0.join()
p1.join()
p2.join()