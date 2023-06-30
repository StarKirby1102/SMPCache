#!/usr/bin/env python3

from pydoc import plain
import sys
import multiprocessing
import numpy as np
import tensorflow as tf
import time
import random

def loadData(filePathList):
    '''
    Load the plaintext from csv/txt to numpy matrix

    This function just for test, use random to simulate the basic demo
    '''
    matrix_0 = np.loadtxt(open(filePathList[0].format(id)), delimiter = ",", skiprows = 1)
    matrix_1 = np.loadtxt(open(filePathList[1].format(id)), delimiter = ",", skiprows = 1)
    matrix_2 = np.loadtxt(open(filePathList[2].format(id)), delimiter = ",", skiprows = 1)

    return [matrix_0, matrix_1, matrix_2]

def generateSS(idlist, plaintext):
    '''
    Encrypt the plaintext to secret share by secureNN protocol

    idlist-which party(s) upload the plaintext
    '''
    import latticex.rosetta as rtt

    '''
    Not every participant will uploads data
    '''

    P_num = len(idlist)
    if P_num == 1:
        rtx = rtt.controller.PrivateDataset(["P{}".format(idlist[0])]).load_X(plaintext[idlist[0]])
    elif P_num == 2:
        # rtx = rtt.controller.PrivateDataset(["P{}".format(idlist[0]), "P{}".format(idlist[1])]).load_X(plaintext[idlist[0]], plaintext[idlist[1]])
        rtx = rtt.controller.PrivateDataset(["P{}".format(idlist[0])]).load_X(plaintext[idlist[0]])
        rtx = np.append(rtx, rtt.controller.PrivateDataset(["P{}".format(idlist[1])]).load_X(plaintext[idlist[1]]), axis = 1)
    else:
        # rtx = rtt.controller.PrivateDataset(["P0", "P1", "P2"]).load_X(plaintext[0], plaintext[1], plaintext[2])
        rtx = rtt.controller.PrivateDataset(["P{}".format(idlist[0])]).load_X(plaintext[idlist[0]])
        rtx = np.append(rtx, rtt.controller.PrivateDataset(["P{}".format(idlist[1])]).load_X(plaintext[idlist[1]]), axis = 1)
        rtx = np.append(rtx, rtt.controller.PrivateDataset(["P{}".format(idlist[2])]).load_X(plaintext[idlist[2]]), axis = 1)

    return rtx

def saveFile(id, ciphertext, saveTableName):
    '''
    Save the data slice to local binary file.

    Note: we store the tuple in ciphertext as S16 type.
    '''
    # filePath = "cache/P{}_".format(id) + saveTableName + "_ciphertext.sdata"
    filePath = "server/P{}/P{}_".format(id,id) + saveTableName + ".sdata"
    tabletobesaved = ciphertext.astype("S16")
    bytesbuffer = tabletobesaved.tobytes()
    fd = open(filePath, "wb")
    fd.write(bytesbuffer)
    fd.close()

    print("Party{} has successully cached to local".format(id))

def saveCipherTable(id, idlist, filePathList, saveTableName):
    '''
    This function simulate the SecureNN save ciphertext table to local binary file:

    Step1. Parties upload the plaintext table
    Step2. Generate the ciphertext table 
    Step3. Store the ciphertext table to local as MPCDB_Cache

    Return a signal indicate that if generate and store successfully, then return true
    '''

    sys.argv.extend(["--node_id", "P{}".format(id)])
    import latticex.rosetta as rtt

    rtt.activate("SecureNN")
    
    plaintext = loadData(filePathList)

    '''
    Define which party(s) will upload the dataset
    '''

    res = generateSS(idlist, plaintext)

    sess = tf.Session()
    sess.run( tf.compat.v1.global_variables_initializer())
    # Take a glance at the ciphertext
    ciphertext = tf.reshape(res, res.shape)
    cipher_result = sess.run(ciphertext)
    # print('From ID:{} local ciphertext result:\n'.format(id), cipher_result)
    
    saveFile(id, cipher_result, saveTableName)

    # Set only party a and c can get plain result
    a_and_c_can_get_plain = 0b101 
    print('From ID:{} plaintext reveal result:\n'.format(id), sess.run(rtt.SecureReveal(cipher_result, a_and_c_can_get_plain)))

# idlist = [0, 1, 2]
# filePathList = ["plaintext/toydataset_P0.csv", "plaintext/toydataset_P1.csv", "plaintext/toydataset_P2.csv"]

# p0 = multiprocessing.Process(target = saveCipherTable, args = (0, idlist, filePathList, "testTable"))
# p1 = multiprocessing.Process(target = saveCipherTable, args = (1, idlist, filePathList, "testTable"))
# p2 = multiprocessing.Process(target = saveCipherTable, args = (2, idlist, filePathList, "testTable"))

# p0.daemon = True
# p0.start()
# p1.daemon = True
# p1.start()
# p2.daemon = True
# p2.start()

# p0.join()
# p1.join()
# p2.join()