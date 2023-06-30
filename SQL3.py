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

    SELECT COUNT(*)
    FROM store_sales INNER JOIN store_returns ON store_sales.PID = store_returns.PID
    WHERE store_returns.ReturnDate - store_sales.SaleDate <= 10

Note: this test just for this SQL, deployer can parse your own SQL to use our MPC-Cache idea
'''

def test_demo3(id):
    sys.argv.extend(["--node_id", "P{}".format(id)])
    import latticex.rosetta as rtt
    rtt.backend_log_to_stdout(False)
    rtt.activate("SecureNN")

    SQL3_START = time.time()

    '''
    Step1. load the ciphertable to P1 and P2
    '''

    if id == 0:
        '''
        Generate the 0, 1's SS
        '''
        plaintext = [[0], [1]]
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(plaintext)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
    elif id == 1:
        test_table = 'users/user2/store_sales_30.csv'
        plaintext = np.loadtxt(open(test_table), delimiter = ",")
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(plaintext)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
    else:
        test_table = 'users/user2/store_returns_30.csv'
        plaintext = np.loadtxt(open(test_table), delimiter = ",")

        '''
        Note: Concat a one-SS column to joinTable in order to compute the COUNT(*) later
        '''
        plaintext = np.hstack((plaintext, np.ones((len(plaintext), 1))))
        # print(len(plaintext[0]))
        # print(plaintext)
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(plaintext)
    
    header = ['SaleDate', 'SaleTime', 'item', 'customer', 'cdemo', 'hdemo', 'addr', 'store', 'promo', 'ticket', 'quantity', 
              'wholeSaleCost', 'listPrice', 'salesPrice', 'extDiscount', 'extwholeSaleCost', 'extlistPrice', 'extsalesPrice', 
              'extTax', 'coupon', 'netPaid', 'netPaidIncTax', 'netProfit', 'zero', 'ReturnDate', 'one']
    print('Successfully load the data')
    
    # y = rtt.RttPlaceholder(tf.float32, shape=rty.shape)
    # z = rtt.RttPlaceholder(tf.float32, shape=rtz.shape)

    MPC_START = time.time()

    sess = tf.Session()
    
    '''
    Step2. join two tables according to PID

    Note: we just use the 2th table's return_date column and one-SS column
    '''
    joinTable = rtt.SecurePsi(rty, rtz, 2, 2, jointlist = [0, 21])

    psi_result = tf.reshape(joinTable[:,25], (-1, 1))

    InnerJoinTable = rtt.SecureMul(joinTable, psi_result)

    # print(sess.run(rtt.SecureReveal(InnerJoinTable)))


    # print(rty.shape)


    # res = rtt.SecureReveal(joinTable)

    # resp = sess.run(res)

    # print(resp)



p0 = multiprocessing.Process(target = test_demo3, args = (0,))
p1 = multiprocessing.Process(target = test_demo3, args = (1,))
p2 = multiprocessing.Process(target = test_demo3, args = (2,))

p0.daemon = True
p0.start()
p1.daemon = True
p1.start()
p2.daemon = True
p2.start()

p0.join()
p1.join()
p2.join()