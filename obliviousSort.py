#!/usr/bin/env python3

import multiprocessing
import sys
import time
import numpy as np
import tensorflow as tf
import random

sessruntotal = 0

def oblivious_cond_swap_bit(cond, x, y):
    '''
    Conditional private swap for single attribute

    Param-cond: compare result(SS) for x and y

    Formula: return cond * x + y - cond * y, cond * y + x - cond * x
    '''
    import latticex.rosetta as rtt

    # sess = tf.Session()
    # cond = sess.run(cond)

    # cx = sess.run(rtt.SecureMul(cond, x))
    # cy = sess.run(rtt.SecureMul(cond, y))

    # tempAdd_1 = sess.run(rtt.SecureAdd(cx, y))
    # tempAdd_2 = sess.run(rtt.SecureAdd(cy, x))

    # res_1 = rtt.SecureSub(tempAdd_1, cy)
    # res_2 = rtt.SecureSub(tempAdd_2, cx)
    res_1 = rtt.SecureAdd(rtt.SecureMul(cond, rtt.SecureSub(x, y)), y)
    res_2 = rtt.SecureAdd(rtt.SecureMul(cond, rtt.SecureSub(y, x)), x)
    # res_1 = rtt.SecureSub(rtt.SecureAdd(rtt.SecureMul(cond, x), y), rtt.SecureMul(cond, y))
    # res_2 = rtt.SecureSub(rtt.SecureAdd(rtt.SecureMul(cond, y), x), rtt.SecureMul(cond, x))
    # res_1 = tf.subtract(tf.add(tf.multiply(cond, x), y), tf.multiply(cond, y))
    # res_2 = tf.subtract(tf.add(tf.multiply(cond, y), x), tf.multiply(cond, x))

    return res_1, res_2

def oblivious_cond_swap_record(attr, InputVector, index_1, index_2, ifDummy, mode):
    '''
    Conditional private swap between records

    e.g. If we want to swap two record according to AGE in ascending order.
    ID    AGE   DEPOSIT             ID    AGE    DEPOSIT
    ......                          ......
    ......                          ......
    1003  38    1500000    --->     1015  25     80000
    ......                          ......
    ......                          ......
    ......                          ......
    1015  25    80000               1003  38     1500000

    attr: The column attribute to sort by. e.g. If we want to sort a table(ID, AGE, DEPOSIT) according to AGE, then attr is 1
    InputVector: The vector to be sort (SS) Note. Should transfer the ciphertext from tf.Tensor to ndarray
    index_1 / index_2: Two objects' indexs to be swapped (plaintext)
    ifDummy: Marking whether each element is dummy -> 0: Real record; 1: Dummy (plaintext)
    mode: Sorting mode -> 0: Ascending; 1: Descending (plaintext)

    '''
    import latticex.rosetta as rtt

    x_value = InputVector[index_1]
    y_value = InputVector[index_2]

    x_flag = ifDummy[index_1]
    y_flag = ifDummy[index_2]

    '''
    Swap strategy: move the dummy to the end of vector and shrink
    '''
    if (x_flag == 1) & (y_flag == 0):
        '''
        If the front element is real value while backend element is dummy, then do swap
        '''
        InputVector[index_1] = y_value
        ifDummy[index_1] = y_flag

        InputVector[index_2] = x_value
        ifDummy[index_2] = x_flag
        
    else:
        if y_flag == 1:
            '''
            If the backend element is dummy, no need to swap
            '''
            pass
        else:
            '''
            If both of x_value and y_value are real value, then do conditional private compare(obliviously swap)
            '''
            x_attr = InputVector[index_1][attr]
            y_attr = InputVector[index_2][attr]
            
            if mode == 0:
                sig = rtt.SecureLess(np.array(x_attr), np.array(y_attr))
            else:
                sig = rtt.SecureGreater(np.array(x_attr), np.array(y_attr))
            
            sess = tf.Session()

            # x_value_tensor_new, y_value_tensor_new = oblivious_cond_swap_bit(sig, np.array(x_value), np.array(y_value))
            # TIME_START = time.time()
            x_value_tensor_new, y_value_tensor_new = oblivious_cond_swap_bit(sig, x_value, y_value)
            # TIME_END = time.time()
            # print('Each obviously swap time:', TIME_END - TIME_START)

            # TIME_START = time.time()
            tf.reset_default_graph()
            x_value_new = sess.run(x_value_tensor_new)
            y_value_new = sess.run(y_value_tensor_new)
            # TIME_END = time.time()
            # global sessruntotal
            # sessruntotal += (TIME_END - TIME_START)
            # print('Each sess.run time:', TIME_END - TIME_START)

            InputVector[index_1] = x_value_new
            InputVector[index_2] = y_value_new

def oblivious_odd_even_merge_sort(attr, InputVector, mode, sorted_length = 1):
    '''
    This module will sort the InputVector according to attr and mode

    Step1. Pads the vector to power of 2 with dummy elements
    '''
    original_length = len(InputVector)
    padding_length = len(InputVector)
    while padding_length & (padding_length - 1) != 0:
        padding_length += 1
    
    record_len = len(InputVector[0])
    
    rd = np.random.RandomState(1487)
    DataSort = []
    DataFlag = []
    for idx in range(padding_length):
        if idx < original_length:
            '''
            Push the real value
            '''
            DataSort.append(InputVector[idx])
            DataFlag.append(0)
        else:
            '''
            Padding with dummy
            '''
            DataSort.append(InputVector[rd.randint(0, original_length)])
            DataFlag.append(1)
    
    DataSort = np.array(DataSort)
    
    '''
    Step2. Execute odd-even merge sort obviously
    '''
    l = sorted_length
    num_keys = len(DataSort)
    while l < num_keys:
        l *= 2
        k = 1
        while k < l:
            k *= 2
            n_outer = num_keys // l
            n_inner = l // k
            n_innermost = 1 if k == 2 else k // 2 - 1

            for i in range(n_outer):
                for j in range(n_inner):
                    base = i * l + j
                    step = l // k
                    if k == 2:
                        index1 = base
                        index2 = base + step
                        oblivious_cond_swap_record(attr, DataSort, index1, index2, DataFlag, mode)
                    else:
                        for i_inner in range(n_innermost):
                            m1 = step + i_inner * 2 * step
                            m2 = m1 + base
                            index1 = m2
                            index2 = m2 + step
                            oblivious_cond_swap_record(attr, DataSort, index1, index2, DataFlag, mode)
    
    DataSort = DataSort[0 : original_length, 0 : record_len]
    # global sessruntotal
    # print('sess.run total time:', sessruntotal)
    return DataSort

# def test(id):
#     sys.argv.extend(["--node_id", "P{}".format(id)])
#     import latticex.rosetta as rtt
#     rtt.backend_log_to_stdout(False)
#     rtt.activate("SecureNN")

#     test_table = 'users/user1/S_user1_table0.csv'

#     TIME_START = time.time()

#     if id == 0:
#         plaintext = np.loadtxt(open(test_table), delimiter = ",", skiprows = 1)
#         rtx = rtt.controller.PrivateDataset(["P0"]).load_X(plaintext)
#         rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
#         rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
#     elif id == 1:
#         rd = np.random.RandomState(1789)
#         plaintext = rd.randint(0, 1, (1, 1))
#         rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
#         rty = rtt.controller.PrivateDataset(["P1"]).load_X(plaintext)
#         rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
#     else:
#         rd = np.random.RandomState(1999)
#         plaintext = rd.randint(0, 1, (1, 1))
#         rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
#         rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
#         rtz = rtt.controller.PrivateDataset(["P2"]).load_X(plaintext)
    
#     session = tf.Session()
#     sorted_rtxdata = oblivious_odd_even_merge_sort(1, rtx, 0)
#     TIME_END = time.time()
#     print('Successfully sort the array, total time:', TIME_END - TIME_START)
#     sorted_rtxdata_plaintext = session.run(rtt.SecureReveal(sorted_rtxdata))
#     tf.get_default_graph().finalize()
#     print(sorted_rtxdata_plaintext)


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