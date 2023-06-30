#!/usr/bin/env python3

import sys
import multiprocessing
import time
import os

def oblivious_cond_swap_bit(cond, x, y):
    '''
    Conditional private swap for single attribute

    Param-cond: compare result(SS) for x and y

    Formula: return cond * x + y - cond * y, cond * y + x - cond * x
    '''
    import latticex.rosetta as rtt

    res_1 = rtt.SecureSub(rtt.SecureAdd(rtt.SecureMul(cond, x), y), rtt.SecureMul(cond, y))
    res_2 = rtt.SecureSub(rtt.SecureAdd(rtt.SecureMul(cond, y), x), rtt.SecureMul(cond, x))

    return res_1, res_2

def SecureOperator(id, op):
    sys.argv.extend(["--node_id", "P{}".format(id)])
    # sys.argv.extend(["--node_id", id])
    import latticex.rosetta as rtt

    import tensorflow as tf
    import numpy as np

    rtt.activate("SecureNN")

    rd = np.random.RandomState(1789)

    x = rd.randint(0, 100, (1, 1))
    y = rd.randint(0, 100, (1, 1))
    z = rd.randint(0, 100, (1, 1))

    dataset0 = [[2, 4, 8], [16, 9, 1], [17, 5, 0], [3, 8, 8], [2, 4, 7], [11, 14, 25]]
    # dataset0 = [[2], [4], [5], [3], [8], [16]]
    dataset1 = [[5, 6, 10], [6, 7, 8], [7, 8, 8], [8, 5, 4], [9, 3, 2], [10, 1, 5]]
    # dataset1 = [[3], [5], [12], [9], [1], [7]]
    dataset2 = [[1], [0], [0], [1], [0], [1]]

    dataset3 = [[6]]
    dataset4 = [[5, 6, 7, 8, 9, 10]]
    # dataset1 = [[4]]
    # dataset2 = [[6]]

    test_table = 'users/user1/S_user1_table0.csv'
    plaintext = np.loadtxt(open(test_table), delimiter = ",", skiprows = 1)

    if id == 0:
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(dataset0)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
    elif id == 1:
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(dataset1)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
    else:
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(dataset2)
    
    # rtx = tf.convert_to_tensor(rtx)
    # rty = tf.convert_to_tensor(rty)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # res = rtt.SecureAdd(rtt.SecureMul(rtz, rtx), rty)

    # res = rtt.SecureSub(rtt.SecureAdd(rtt.SecureMul(rtz, rtx), rty), rty)
    res = rtt.SecureAdd(rtt.SecureMul(rtz, rtt.SecureSub(rtx, rty)), rty)
    for i in range(10000):
        print('Round{}'.format(i + 1))
        sess.run(res)

    # res = rtt.SecureSub(rtt.SecureAdd(rtt.SecureMul(rtz, rtx), rty), rtt.SecureMul(rtz, rty))

    print(sess.run(rtt.SecureReveal(res)))
    # for i in range(len(rtx)):
    #     print('{}th oblivious swap result:'.format(i), sess.run(rtt.SecureReveal(oblivious_cond_swap_bit(rtz[i], rtx[i], rty[i]))))

    # ciphertext = sess.run(cresult)

    # print('From ID:{} plaintext result:\n'.format(id), sess.run(rtt.SecureReveal(ciphertext)))

p0 = multiprocessing.Process(target = SecureOperator, args = (0, "op1"))
p1 = multiprocessing.Process(target = SecureOperator, args = (1, "op1"))
p2 = multiprocessing.Process(target = SecureOperator, args = (2, "op1"))

p0.daemon = True
p0.start()
p1.daemon = True
p1.start()
p2.daemon = True
p2.start()

p0.join()
p1.join()
p2.join()
