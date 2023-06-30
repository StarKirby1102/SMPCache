import numpy as np
import csv
import random
from parameters import *

def genTestdata():
    '''
    This module generate a financial scene dataset for 4 parties

    Party0: Upload the users' ID and AGE
    Party1: Upload the users' credit_rating1 and loan1
    Party2: Upload the users' credit_rating2 and loan2
    Party3: Upload the users' deposit and credit_rating3

    Note: this module just generate a toy dataset for financial scene
    '''
    rd = np.random.RandomState(1999)
    matrix_0 = []
    matrix_1 = []
    matrix_2 = []
    matrix_3 = []

    param = parameters()

    for i in range(param.dataScale):
        kv0 = [100000+i, rd.randint(18, 80)]
        matrix_0.append(kv0)
    for i in range(param.dataScale):
        kv1 = [100000+i, rd.randint(100, 100000)*100, rd.randint(100, 100000)*100, rd.randint(1, 10)]
        matrix_1.append(kv1)
    for i in range(param.dataScale):
        kv2 = [100000+i, rd.randint(100, 100000)*100, rd.randint(100, 100000)*100, rd.randint(1, 10)]
        matrix_2.append(kv2)
    for i in range(param.dataScale):
        kv3 = [100000+i, rd.randint(100, 100000)*100, rd.randint(100, 100000)*100, rd.randint(1, 10)]
        matrix_3.append(kv3)

    filePath_P0 = "users/user0/S_user0_table0.csv"
    filePath_P1 = "users/user1/S_user1_table0.csv"
    filePath_P2 = "users/user2/S_user2_table0.csv"
    filePath_P3 = "users/user3/S_user3_table0.csv"

    '''
    Generate the plaintext table header
    '''
    header_0 = [["ID", "AGE"]]
    header_1 = [["ID", "deposit1", "loan1", "credit1"]]
    header_2 = [["ID", "deposit2", "loan2", "credit2"]]
    header_3 = [["ID", "deposit3", "loan3", "credit3"]]

    matrix_0 = np.concatenate([header_0, matrix_0], axis = 0)
    matrix_1 = np.concatenate([header_1, matrix_1], axis = 0)
    matrix_2 = np.concatenate([header_2, matrix_2], axis = 0)
    matrix_3 = np.concatenate([header_3, matrix_3], axis = 0)

    '''
    Write the dataset to the csv
    '''

    np.savetxt(filePath_P0, matrix_0, delimiter = ',', fmt="%s")
    np.savetxt(filePath_P1, matrix_1, delimiter = ',', fmt="%s")
    np.savetxt(filePath_P2, matrix_2, delimiter = ',', fmt="%s")
    np.savetxt(filePath_P3, matrix_3, delimiter = ',', fmt="%s")


if __name__ == '__main__':
    # main()
    genTestdata()