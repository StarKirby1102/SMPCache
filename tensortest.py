import multiprocessing
import sys
import tensorflow as tf

def sort_test(id):
    sys.argv.extend(["--node_id", "P{}".format(id)])
    # sys.argv.extend(["--node_id", id])
    import latticex.rosetta as rtt
    import numpy as np

    rtt.activate("SecureNN")
    test_table = 'users/user1/S_user1_table0.csv'

    if id == 0:
        plaintext = np.loadtxt(open(test_table), delimiter = ",", skiprows = 1)
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(plaintext)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
    elif id == 1:
        b = [[0]]
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(b)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
    else:
        c = [[0]]
        rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
        rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
        rtz = rtt.controller.PrivateDataset(["P2"]).load_X(c)
    
    rtx = tf.convert_to_tensor(rtx)
    # a = tf.sort(rtx)
    # sess = tf.Session()
    # sa = sess.run(a)
    # print(sess.run(rtt.SecureReveal(sa)))
    sess = tf.Session()
    for i in range(10000):
        arr = sess.run(rtx)
    print(type(arr))


p0 = multiprocessing.Process(target = sort_test, args = (0,))
p1 = multiprocessing.Process(target = sort_test, args = (1,))
p2 = multiprocessing.Process(target = sort_test, args = (2,))

p0.daemon = True
p0.start()
p1.daemon = True
p1.start()
p2.daemon = True
p2.start()

p0.join()
p1.join()
p2.join()


