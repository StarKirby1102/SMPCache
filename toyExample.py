#!/usr/bin/env python3
from pydoc import plain
import sys
import multiprocessing
import numpy as np
import tensorflow as tf
import time
import random

from saveCipherTable import saveCipherTable
from loadCipherTable import loadCipherTable
from queryParse import parseSQL
from executeSecurePlan import executeSecurePlan

def toyExample(id, idlist, plaintextlist, tableName, tableShape, SQL):

    saveCipherTable(id, idlist, plaintextlist, tableName)

    loadCipherTable(id, tableName, tableShape)

    # plan = parseSQL(SQL)

    # executeSecurePlan([1,0,1,0], plan)

# if __name__ == '__main__':
#     plan = parseSQL("SELECT deposit,credit1 FROM tableA WHERE deposit>200000 AND deposit<1000000")
#     executeSecurePlan([1,0,1,0], plan)

idlist = [0, 1, 2]
filePathList = ["plaintext/toydataset_P0.csv", "plaintext/toydataset_P1.csv", "plaintext/toydataset_P2.csv"]
testSQL = "SELECT deposit,credit1 FROM tableA WHERE deposit>200000 AND deposit<1000000"

p0 = multiprocessing.Process(target = toyExample, args = (0, idlist, filePathList, "toyExampleTable", [1000, 6], testSQL))
p1 = multiprocessing.Process(target = toyExample, args = (1, idlist, filePathList, "toyExampleTable", [1000, 6], testSQL))
p2 = multiprocessing.Process(target = toyExample, args = (2, idlist, filePathList, "toyExampleTable", [1000, 6], testSQL))

p0.daemon = True
p0.start()
p1.daemon = True
p1.start()
p2.daemon = True
p2.start()

p0.join()
p1.join()
p2.join()