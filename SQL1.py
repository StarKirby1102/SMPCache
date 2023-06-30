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
# from secureGroupBy import *
from cache import *
from parameters import *
from executeSecurePlan import *

'''
In this py file, we test the SQL demo on TPC-H lineitem(part):

    SELECT
        sum(l_extendedprice * l_discount) as revenue
    FROM
        lineitem
    WHERE
        (l_shipdate >= 1995-01-01 AND l_shipdate < 1998-12-01)
        AND (l_discount >= 0.05 AND l_discount <= 0.07)
        AND (l_quantity < 24)

Note: this test just for this SQL, deployer can parse your own SQL to use our MPC-Cache idea
'''

def test_demo1(id):
    sys.argv.extend(["--node_id", "P{}".format(id)])
    import latticex.rosetta as rtt
    rtt.backend_log_to_stdout(False)
    rtt.activate("SecureNN")

    sess = tf.Session()

    SQL1_START = time.time()

    header = ['l_orderkey', 'l_partkey', 'l_suppkey', 'l_linenumber', 'l_quantity', 'l_extendedprice', 'l_discount', 'l_tax', 'l_shipdate', 'l_commitdate', 'l_receiptdate']
    SQL1_START = time.time()

    '''
    Step1. select the record that   (l_shipdate >= 1995-01-01 AND l_shipdate < 1998-12-01)
                                    AND (l_discount >= 0.05 AND l_discount <= 0.07)
                                    AND (l_quantity < 24)

    '''

    SQL_SUB1 = [['l_shipdate', 'op3', '19950101'], 'AND', ['l_shipdate', 'op1', '19981201']]

    SQL_SUB2 = [['l_discount', 'op3', '0.05'], 'AND', ['l_discount', 'op1', '0.07']]

    SQL_SUB3 = [['l_quantity', 'op1', '24']]

    opAST1 = infix2postfix(SQL_SUB1)

    opAST2 = infix2postfix(SQL_SUB2)

    opAST3 = infix2postfix(SQL_SUB3)


    if id == 0:
        '''
        Generate the 1's SS
        '''
        plaintext = [[0], [1]]
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(plaintext)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
    elif id == 1:
        test_table = 'users/user1/lineitem_2097152.csv'
        plaintext = np.loadtxt(open(test_table), delimiter = ",", skiprows = 1)
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(plaintext)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
    else:
        plaintext = [[0], [1]]
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(plaintext)
    
    print('Successfully load the data')

    MPC_START = time.time()

    where_result_1 = computePostfix(id, opAST1, rty, header)
    where_result_1 = tf.transpose(where_result_1)

    where_result_2 = [computePostfix(id, opAST2, rty, header)]
    where_result_2 = tf.transpose(where_result_2)

    where_result_3 = computePostfix(id, opAST3, rty, header)
    where_result_3 = tf.transpose(where_result_3)

    where_result = rtt.SecureLogicalAnd((rtt.SecureLogicalAnd(where_result_1, sess.run(where_result_2))), sess.run(where_result_3))


    # condition_rtdata = rtt.SecureMul(where_result_1, rty)

    # print(sess.run(rtt.SecureReveal(where_result_1)))

    # print(sess.run(rtt.SecureReveal(where_result_2)))

    # print(sess.run(rtt.SecureReveal(where_result_3)))

    result = rtt.SecureMul(where_result, rty)

    '''
    Step2. compute the sum(l_extendedprice * l_discount)
    '''
    result = sess.run(result)
    
    revenue = rtt.SecureMul(result[:,5], result[:,6])
    ssum = rtt.SecureSum(revenue)
    # print(sess.run(rtt.SecureReveal(ssum)))
    # ssum = rtx[0]
    # for i in range(len(result)):
    #     # tf.reset_default_graph()
    #     # print(i)
    #     ssum = rtt.SecureAdd(ssum, rtt.SecureMul(result[i][5], result[i][6]))
    #     ssum = sess.run(ssum)
    # print(sess.run(rtt.SecureReveal(ssum)))

    SQL1_END = time.time()

    print('MPC time:', SQL1_END - MPC_START)

    print('Total time:', SQL1_END - SQL1_START)


    # print(sess.run(rtt.SecureReveal(result)))


    # print(sess.run(rtt.SecureReveal(sorted_rtxdata)))
    # print(sess.run(rtt.SecureReveal(compare_result)))
    # print(sess.run(rtt.SecureReveal(D_0)))
    # print(sess.run(rtt.SecureReveal(C_)))
    # print(sess.run(rtt.SecureReveal(D_1)))

p0 = multiprocessing.Process(target = test_demo1, args = (0,))
p1 = multiprocessing.Process(target = test_demo1, args = (1,))
p2 = multiprocessing.Process(target = test_demo1, args = (2,))

p0.daemon = True
p0.start()
p1.daemon = True
p1.start()
p2.daemon = True
p2.start()

p0.join()
p1.join()
p2.join()