#!/usr/bin/env python3

from pydoc import plain
import sys
import multiprocessing
import numpy as np
import tensorflow as tf
import time
import random

def loadCipherTable(id, saveTableName, cipherTableShape):
    '''
    This module read and reconstruct the ciphertext table from bytesbuffer
    '''
    # filePath = "cache/P{}_".format(id) + saveTableName + "_ciphertext.sdata"
    filePath = "server/P{}/P{}_".format(id,id) + saveTableName + ".sdata"
    fd = open(filePath, "rb")
    bytesbuffer = fd.read()
    fd.close()
    '''
    The reconstruct parameters is equal to saveCipherTable.py
    '''
    cipherTable = np.frombuffer(bytesbuffer, dtype = "S16", count = cipherTableShape[0]*cipherTableShape[1], offset = 0)
    cipherTable.resize((cipherTableShape[0], cipherTableShape[1]))
    return cipherTable

def plaintextReveal(id, saveTableName, cipherTableShape):
    '''
    This module aimd to verify the correctness for loadCipherTable module
    '''
    sys.argv.extend(["--node_id", "P{}".format(id)])
    import latticex.rosetta as rtt
    rtt.activate("SecureNN")

    res = loadCipherTable(id, saveTableName, cipherTableShape)
    sess = tf.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    # Take a glance at the ciphertext
    ciphertext = tf.reshape(res, res.shape)
    cipher_result = sess.run(ciphertext)
    # print('From ID:{} local ciphertext result:\n'.format(id), cipher_result)

    a_and_c_can_get_plain = 0b101 
    print('From ID:{} plaintext reveal result:\n'.format(id), sess.run(rtt.SecureReveal(cipher_result, a_and_c_can_get_plain)))


# p0 = multiprocessing.Process(target = plaintextReveal, args = (0, "toyExampleTable", [1000, 6]))
# p1 = multiprocessing.Process(target = plaintextReveal, args = (1, "toyExampleTable", [1000, 6]))
# p2 = multiprocessing.Process(target = plaintextReveal, args = (2, "toyExampleTable", [1000, 6]))

# p0.daemon = True
# p0.start()
# p1.daemon = True
# p1.start()
# p2.daemon = True
# p2.start()

# p0.join()
# p1.join()
# p2.join()