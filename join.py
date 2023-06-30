#!/usr/bin/env python3

import sys
import multiprocessing
import numpy as np

def doprocess(id,op,data):

    import time

    curtime = time.time()
    sys.argv.extend(["--node_id", "P{}".format(id)])
    import latticex.rosetta as rtt

    import tensorflow as tf
    import numpy as np

    rtt.activate("SecureNN")
    print(time.time() - curtime)
    curtime = time.time()

    #a = np.random.randint(low=1, high=20, size=10)
    #a = np.random.choice(10,5,replace=False)
    a= np.ndarray((8,1),buffer=np.array([1,2,3,4,5,600,18,70]),dtype=int)
    
    if id == 0:
        print(a)
    #b = a.copy()
    b = np.concatenate( [np.ndarray((8,1),buffer=np.array([700,6,18,4,5,3,45,2]),dtype=int), np.random.choice(30,8,replace=False).reshape((-1, 1))], axis=1)
    b = np.concatenate( [b, np.random.choice(30,8,replace=False).reshape((-1, 1))], axis=1)

    #np.random.shuffle(b)
    if id == 0:
        print(b)

    if id == 0:
        rta = rtt.controller.PrivateDataset(["P0"]).load_X(a)
    else:
        rta = rtt.controller.PrivateDataset(["P0"]).load_X(None)
    if id == 1:
        rtb = rtt.controller.PrivateDataset(["P1"]).load_X(b)
    else:
        rtb = rtt.controller.PrivateDataset(["P1"]).load_X(None)

    x = rtt.RttPlaceholder(tf.float32, shape=rta.shape)
    y = rtt.RttPlaceholder(tf.float32, shape=rtb.shape)
    z = rtt.SecurePsi(x, y, jointlist=[0, 2])

    a_and_c_can_get_plain = None
    res = rtt.SecureReveal(z, a_and_c_can_get_plain)
    # Start execution
    sess = tf.Session()
    # print("~~~~~~~~~~~~~~~~~~~~~~~DAGtime", time.time() - curtime)
    curtime = time.time()
    resp = sess.run(res, feed_dict={x: rta, y:rtb})

    print("~~~~~~~~~~~~~~~~~~~~~~~runtime", time.time() - curtime)
    if id == 0:
        print(resp)

op=0
data=10

p0 = multiprocessing.Process(target=doprocess, args=(0,op,data))
p1 = multiprocessing.Process(target=doprocess, args=(1,op,data))
p2 = multiprocessing.Process(target=doprocess, args=(2,op,data))

p0.daemon = True
p0.start()
p1.daemon = True
p1.start()
p2.daemon = True
p2.start()


p0.join()
p1.join()
p2.join()

