#!/usr/bin/env python3

import multiprocessing
import sys
import time
import numpy as np
import tensorflow as tf
import csv
import os
from parameters import *
from obliviousSort import *

def secureGroupBy(ciphertable, attr = 0, having = None):
    '''
    This module will do the secure group by
    '''

    '''
    Step1. sort the ciphertable according to attr
    '''
    import latticex.rosetta as rtt

    '''
    Step2. compute the SS of compare result
    '''
    compare_result = []
    sess = tf.Session()
    sorted_table = oblivious_odd_even_merge_sort(attr, ciphertable, 0)
    for i in range(len(sorted_table) - 1):
        compare_result.append(sess.run(rtt.SecureEqual(tf.convert_to_tensor([sorted_table[i][attr]]), tf.convert_to_tensor([sorted_table[i+1][attr]]))))
    compare_result = np.array(compare_result)
    # print(sess.run(rtt.SecureReveal(compare_result)))
    

    return sorted_table, compare_result

def test(id):
    sys.argv.extend(["--node_id", "P{}".format(id)])
    import latticex.rosetta as rtt
    rtt.backend_log_to_stdout(False)
    rtt.activate("SecureNN")

    test_table = 'users/user0/S_user0_test.csv'

    TIME_START = time.time()

    if id == 0:
        plaintext = np.loadtxt(open(test_table), delimiter = ",", skiprows = 1)
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
    group_rtxdata = secureGroupBy(rtx, 0, None)
    TIME_END = time.time()
    print('Successfully group the array, total time:', TIME_END - TIME_START)
    print('ciphertext:\n', group_rtxdata)
    group_rtxdata_plaintext = session.run(rtt.SecureReveal(group_rtxdata))
    tf.get_default_graph().finalize()
    print(group_rtxdata_plaintext)


# p0 = multiprocessing.Process(target = test, args = (0,))
# p1 = multiprocessing.Process(target = test, args = (1,))
# p2 = multiprocessing.Process(target = test, args = (2,))

# p0.daemon = True
# p0.start()
# p1.daemon = True
# p1.start()
# p2.daemon = True
# p2.start()

# p0.join()
# p1.join()
# p2.join()