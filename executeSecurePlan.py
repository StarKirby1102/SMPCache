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
from cache import *
from parameters import *

def judgeTypes(tuple):
    '''
    When compute the result from the operand stack, we should judge the type of the operand:

    There are three types:
    (1) Digit: e.g. '4', '50000'        return 1
    (2) Column: e.g. 'ID', 'deposit3'            return 2
    (3) Cipher result: e.g. [[b'ul\xcd\x11\x89\x9d\xba\xe1#'][b'\x9fIa\xcbj%8n#']...]       return 3

    '''
    if type(tuple) != str:
        return 3
    else:
        if tuple.isdigit() == True:
            return 1
        else:
            return 2

def computePostfix(id, postexp, cipherTable, header):
    '''
    This function will compute the result according to postfix expressions
    '''
    import latticex.rosetta as rtt
    # rtt.activate("SecureNN")
    operand = []

    # i = 0
    flag = 0
    for token in postexp:
        if token not in ['ALL', 'ANY', 'BETWEEN', 'LIKE', 'IN', 'OR', 'SOME', 'AND', 'NOT', 'op0', 'op1', 'op2', 'op3', 'op4', 'op5', 'op6', 'op7', 'op8', 'op9', '(', ')']:
            '''
            If token is operand, push to the temp stack
            '''
            operand.append(token)
        else:
            '''
            If token is operator, compute the intermediate results and push to the stack
            '''
            roperand = operand.pop()
            loperand = operand.pop()

            '''
            Judge the operand type and generate the operand(SS):
            If operand is digit, generate the share of digit directly
            If operand is string, select the cipher column from the ciphertable
            If operand is ndarray, use it directly
            '''
            param = parameters()
            # cacheFlag = cacheRule(id, loperand, roperand, token) and param.cacheTurn
            cacheFlag = param.cacheTurn
            if cacheFlag == True:
                '''
                If the table has cached?

                True -> load the MPC cache directly;
                False -> execute MPC
                '''
                cachePath = cacheName(loperand, roperand, token)
                if ifCached(id, cachePath) == True and param.cacheTurn == True:
                    operand.append(cacheLoad(id, cachePath, [1, param.dataScale]))
                else:
                    for cidx in range(len(header)):
                        if header[cidx] == loperand:
                            loperand = cipherTable[:,cidx]
                            break
                        else:
                            continue

                    for cidx in range(len(header)):
                        if header[cidx] == roperand:
                            roperand = cipherTable[:,cidx]
                            break
                        else:
                            continue
                    
                    rttres = secureOperator(id, loperand, roperand, token)
                    '''
                    Judge if the rttres need to be cached
                    '''
                    cacheSave(id, rttres, cachePath)

                    operand.append(rttres)

            else:
                if judgeTypes(loperand) == 1:
                    loperand = [[int(loperand)]]
                    if id == 0:
                        rtdata0 = rtt.controller.PrivateDataset(["P0"]).load_X(loperand)
                        rtdata1 = rtt.controller.PrivateDataset(["P1"]).load_X(None)
                        rtdata2 = rtt.controller.PrivateDataset(["P2"]).load_X(None)
                    if id == 1:
                        rtdata0 = rtt.controller.PrivateDataset(["P0"]).load_X(None)
                        rtdata1 = rtt.controller.PrivateDataset(["P1"]).load_X(loperand)
                        rtdata2 = rtt.controller.PrivateDataset(["P2"]).load_X(None)
                    if id == 2:
                        rtdata0 = rtt.controller.PrivateDataset(["P0"]).load_X(None)
                        rtdata1 = rtt.controller.PrivateDataset(["P1"]).load_X(None)
                        rtdata2 = rtt.controller.PrivateDataset(["P2"]).load_X(loperand)
                    loperand = rtdata0
                    # loperand = tf.broadcast_to(rtdata0, [20,1])

                elif judgeTypes(loperand) == 2:
                    for cidx in range(len(header)):
                        if header[cidx] == loperand:
                            loperand = cipherTable[:,cidx]
                            break
                        else:
                            continue

                if judgeTypes(roperand) == 1:
                    roperand = [[int(roperand)]]
                    if id == 0:
                        rtdata0 = rtt.controller.PrivateDataset(["P0"]).load_X(roperand)
                        rtdata1 = rtt.controller.PrivateDataset(["P1"]).load_X(None)
                        rtdata2 = rtt.controller.PrivateDataset(["P2"]).load_X(None)
                    if id == 1:
                        rtdata0 = rtt.controller.PrivateDataset(["P0"]).load_X(None)
                        rtdata1 = rtt.controller.PrivateDataset(["P1"]).load_X(roperand)
                        rtdata2 = rtt.controller.PrivateDataset(["P2"]).load_X(None)
                    if id == 2:
                        rtdata0 = rtt.controller.PrivateDataset(["P0"]).load_X(None)
                        rtdata1 = rtt.controller.PrivateDataset(["P1"]).load_X(None)
                        rtdata2 = rtt.controller.PrivateDataset(["P2"]).load_X(roperand)
                    roperand = rtdata0
                    # roperand = tf.broadcast_to(rtdata0, [20,1])

                elif judgeTypes(roperand) == 2:
                    for cidx in range(len(header)):
                        if header[cidx] == roperand:
                            roperand = cipherTable[:,cidx]
                            break
                        else:
                            continue
                
                rttres = secureOperator(id, loperand, roperand, token)
                '''
                Judge if the rttres need to be cached
                '''
                operand.append(rttres)

    return operand.pop()

def secureOperator(id, leftOperand, rightOperand, op):
    import latticex.rosetta as rtt
    # rtt.activate("SecureNN")
    '''
    After generate the rtt operand, this module will compute the result
    Operator   ReplaceCode
    >          op0
    <          op1
    ==         op2
    >=         op3
    <=         op4
    <>         op5
    +          op6
    -          op7
    *          op8
    /          op9
    '''

    if op == 'op0':
        rtres = rtt.SecureGreater(leftOperand, rightOperand)
    elif op == 'op1':
        rtres = rtt.SecureLess(leftOperand, rightOperand)
    elif op == 'op2':
        rtres = rtt.SecureEqual(leftOperand, rightOperand)
    elif op == 'op3':
        rtres = rtt.SecureGreaterEqual(leftOperand, rightOperand)
    elif op == 'op4':
        rtres = rtt.SecureLessEqual(leftOperand, rightOperand)
    elif op == 'op5':
        rtres = rtt.SecureLogicalNot(rtt.SecureEqual(leftOperand, rightOperand))
    elif op == 'op6':
        rtres = rtt.SecureAdd(leftOperand, rightOperand)
    elif op == 'op7':
        rtres = rtt.SecureSub(leftOperand, rightOperand)
    elif op == 'op8':
        rtres = rtt.SecureMul(leftOperand, rightOperand)
    elif op == 'op9':
        rtres = rtt.SecureSecureFloorDiv(leftOperand, rightOperand)
    elif op == 'AND':
        rtres = rtt.SecureLogicalAnd(leftOperand, rightOperand)
    elif op == 'OR':
        rtres = rtt.SecureLogicalOr(leftOperand, rightOperand)
    
    # rtres = tf.transpose(rtres)
    return rtres

def executeSecurePlan(id, onlinelist, SQLs):
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    SQLidx = 0
    TIME_START = time.time()
    param = parameters()
    for SQL in SQLs:
        plan = parseSQL(SQL)
        '''
        This module will execute the secure plan generated by queryParse.py

        Parameters:
        (1) onlinelist: Indicate the which user(s) is(are) online/offline. 0 represents offline and 1 represents online
        (2) plan: The key-value datastructure generated by queryParse.py
        '''

        '''
        Step1. This module should identify which column to be selected
        '''
        columns = []
        for item in plan['SELECT']:
            if item[0] not in [',', '(', ')']:
                columns.append(item[0])

        '''
        Step2. Identify which table(plaintext) shoule be uploaded and selected

        table: attributes_source_name

        attribute: S/M, S-single, M-merge
        source: user0/user1/user2/user3/...
        name: table name

        e.g. S_user2_table0
        '''
        table = plan['FROM'][0][0]
        table_attribute = table.split('_')[0]
        table_source = table.split('_')[1]
        table_name = table.split('_')[2]

        tablePath = 'users/{}/{}.csv'.format(table_source, table)
        if os.path.exists(tablePath) == False:
            print('Please check the tablename in SQL!')
            exit(0)

        '''
        Step3. According to header, cipher_result and column, server will return the expected columns of ciphertext table

        Datastructures: 
        (1)table header is stored in header[], the data is recorded in plaintext[]
        (2)select_column_idx: the columns index appear in 'SELECT' key
        (3)where_column_idx: the columns index appear in 'WHERE' key
        (4)cipher_column_idx: the union of select_column_idx and where_column_idx
        '''
        sys.argv.extend(["--node_id", "P{}".format(id)])
        import latticex.rosetta as rtt
        rtt.backend_log_to_stdout(False)

        rtt.activate("SecureNN")

        # R_START = time.time()

        header = []
        with open(tablePath, 'r') as f:
            reader = csv.reader(f)
            header = list(reader)[0]
        header = np.array(header)
    
        # R_END = time.time()
        # print('Read file total time:', R_END - R_START, 's')

        # header = ['ss_sold_date_sk','ss_sold_time_sk','ss_item_sk','ss_customer_sk','ss_cdemo_sk','ss_hdemo_sk','ss_addr_sk','ss_store_sk','ss_promo_sk','ss_ticket_number','ss_quantity','ss_wholesale_cost','ss_list_price','ss_sales_price','ss_ext_discount_amt','ss_ext_sales_price','ss_ext_wholesale_cost','ss_ext_list_price','ss_ext_tax','ss_coupon_amt','ss_net_paid','ss_net_paid_inc_tax','ss_net_profit','ss_null_col']
        # header = np.array(header)

        select_column_idx = []
        for column in columns:
            for idx in range(len(header)):
                if column == header[idx]:
                    select_column_idx.append(idx)
                else:
                    continue
        
        select_delete_columns = []
        if columns[0] == '*':
            select_delete_columns = []
        else:
            for i in range(len(header)):
                if i in select_column_idx:
                    continue
                else:
                    select_delete_columns.append(i)
        

        '''
        Parse and compute the 'ORDER BY' subquery

        Note. This module just support single attribute sort

        e.g. 'SELECT ID,AGE FROM TABLE ORDER BY DEPOSIT ASC|DESC'
        '''
        if 'ORDER BY' in plan.keys():
            order_columns = []
            order_column_idx = []
            # if 'ORDER BY' in plan.keys():
            kv = plan['ORDER BY'][0][0].split(' ')
            order_columns.append(kv[0])
            order_mode = kv[1]

            for column in order_columns:
                for idx in range(len(header)):
                    if column == header[idx]:
                        order_column_idx.append(idx)
                    else:
                        continue

            cipher_column_idx = list(set(select_column_idx).union(set(order_column_idx)))
            # print('order_columns_idx:', order_column_idx, 'select_column_idx:', select_column_idx, 'cipher_column_idx:', cipher_column_idx)
            cipher_delete_column_idx = []
            for i in range(len(header)):
                    if i in cipher_column_idx:
                        continue
                    else:
                        cipher_delete_column_idx.append(i)
            
            '''
            Generate the cache path and judge whether to save/load to cache

            e.g. 'SELECT ID FROM TABLE ORDER BY DEPOSIT ASC' will be cached to server/P*/TABLE_ID_DEPOSIT_ASC.sdata
            '''
            cacheName = table + '_'
            for i in range(len(select_column_idx)):
                cacheName += (header[i] + '_')
            cacheName += (kv[0] + '_' + kv[1])

            if ifCached(id, cacheName) == True and param.cacheTurn == True:
                LOAD_START = time.time()
                rtres = cacheLoad(id, cacheName, [param.dataScale, len(select_column_idx)])
                LOAD_END = time.time()
                print('Cache total time:', LOAD_END - LOAD_START, 's')
            
            else:            
                '''
                StepO-1. We define P0 to generate the data slices for plaintext

                After generating the secret share, P0,P1,P2 will hold the data slices

                Note: P1 and P2 generate random 1*1 matrix to ensure the socket run, just P0 generate the valid secret share
                '''

                if id == 0:
                    plaintext = np.loadtxt(open(tablePath, encoding = 'utf-8'), str, delimiter = ",", skiprows = 1)
                    plaintext = np.delete(plaintext, cipher_delete_column_idx, axis = 1)[0:param.dataScale]
                    '''
                    NULL value handling
                    '''
                    plaintext[np.where(plaintext == '')] = '0'

                    rtx = rtt.controller.PrivateDataset(["P0"]).load_X(plaintext)
                    rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
                    rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
                elif id == 1:
                    rd = np.random.RandomState(1789)
                    plaintext = rd.randint(0, 1, (1, 1))
                    rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
                    rty = rtt.controller.PrivateDataset(["P1"]).load_X(plaintext)
                    rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
                else:
                    rd = np.random.RandomState(1999)
                    plaintext = rd.randint(0, 1, (1, 1))
                    rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
                    rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
                    rtz = rtt.controller.PrivateDataset(["P2"]).load_X(plaintext)

                header = np.delete(header, cipher_delete_column_idx)

                '''
                StepO-2. Execute oblivious odd even merge sort on cipher table

                order_mode: Check whether the sort mode is in ASC|DESC

                attr_idx: Sorting will be done according to attr
                '''
                if order_mode == 'ASC':
                    sortMode = 0
                elif order_mode == 'DESC':
                    sortMode = 1
                else:
                    print('Please check your order mode, just support ASC|DESC!')
                    exit(-1)

                attr_idx = 0
                for attr_idx in range(len(header)):
                    '''
                    Note. This module just support single attribute for sorting
                    '''
                    if header[attr_idx] == order_columns[0]:
                        break
                    else:
                        attr_idx += 1

                SORT_START = time.time()
                
                rtres = oblivious_odd_even_merge_sort(attr_idx, rtx, sortMode)

                SORT_END = time.time()
                print('MPC sort total time:', SORT_END - SORT_START, 's')
                # session = tf.Session()
                # sorted_rtxdata_plaintext = session.run(rtt.SecureReveal(rtres))
                # print(sorted_rtxdata_plaintext)  

                if ifCached(id, cacheName) == False and param.cacheTurn == True:
                    cacheSave(id, rtres, cacheName)

        # session = tf.Session()
        # sorted_rtxdata_plaintext = session.run(rtt.SecureReveal(rtres))
        # print(sorted_rtxdata_plaintext)            


        '''
        Parse and compute the 'WHERE' subquery
        '''
        if 'WHERE' in plan.keys():
            where_columns = []
            where_column_idx = []
            # if 'WHERE' in plan.keys():
            for item in plan['WHERE']:
                for tuple in item:
                    where_columns.append(tuple)
                # print(where_columns)

            for column in where_columns:
                for idx in range(len(header)):
                    if column == header[idx]:
                        where_column_idx.append(idx)
                    else:
                        continue

            cipher_column_idx = list(set(select_column_idx).union(set(where_column_idx)))
            cipher_delete_column_idx = []
            for i in range(len(header)):
                    if i in cipher_column_idx:
                        continue
                    else:
                        cipher_delete_column_idx.append(i)
            
            '''
            StepW-1. We define P0 to generate the data slices for plaintext

            After generating the secret share, P0,P1,P2 will hold the data slices

            Note: P1 and P2 generate random 1*1 matrix to ensure the socket run, just P0 generate the valid secret share
            '''

            if id == 0:
                plaintext = np.loadtxt(open(tablePath, encoding = 'utf-8'), str, delimiter = ",", skiprows = 1, usecols = cipher_column_idx)[0:param.dataScale]
                # plaintext = np.delete(plaintext, cipher_delete_column_idx, axis = 1)[0:param.dataScale]
                '''
                NULL value handling
                '''
                plaintext[np.where(plaintext == '')] = '0'
                rtx = rtt.controller.PrivateDataset(["P0"]).load_X(plaintext)
                rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
                rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
            elif id == 1:
                rd = np.random.RandomState(1789)
                plaintext = rd.randint(0, 1, (1, 1))
                rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
                rty = rtt.controller.PrivateDataset(["P1"]).load_X(plaintext)
                rtz = rtt.controller.PrivateDataset(["P2"]).load_X(None)
            else:
                rd = np.random.RandomState(1999)
                plaintext = rd.randint(0, 1, (1, 1))
                rtx = rtt.controller.PrivateDataset(["P0"]).load_X(None)
                rty = rtt.controller.PrivateDataset(["P1"]).load_X(None)
                rtz = rtt.controller.PrivateDataset(["P2"]).load_X(plaintext)

            header = np.delete(header, cipher_delete_column_idx)
            
            '''
            StepW-2. After generating the whole ciphertext table, we should parse the WHERE subquery

            This module aimed to transfer the WHERE subquery to logic AST
            '''    
            
            opAST = infix2postfix(plan['WHERE'])
            # print(opAST)
            
            select_ans_colidx = []
            select_ans = []

            '''
            Choose the index of 'select' subquery
            '''
            temp = 0
            for idx in range(len(columns)):
                if columns[idx] == header[temp]:
                    select_ans_colidx.append(temp)
                    temp += 1
                else:
                    continue    
            
            sess = tf.compat.v1.Session()
            sess.run(tf.compat.v1.global_variables_initializer())

            WHERE_TIME_START = time.time()

            where_result = computePostfix(id, opAST, rtx, header)

            WHERE_TIME_END = time.time()

            print('MPC time:', WHERE_TIME_END - WHERE_TIME_START, 's')
            # where_result_new = tf.reshape(where_result[0], [param.dataScale, -1])

            # print('WHERE result:', sess.run(rtt.SecureReveal(where_result)))
            # print('rtx result:', sess.run(rtt.SecureReveal(rtx)))

            print('where_result\'s type:', type(where_result), 'rtx\'s type:', type(rtx))
            SQLidx += 1
            print('From ID:{} the {}th SQL\'s parse completed!'.format(id, SQLidx))


    TIME_END = time.time()
    print('Successfully execute all the SQLs, total time:', TIME_END - TIME_START, 's')
    rtt.deactivate()


# SQLs = ['SELECT ID FROM S_user3_table0 WHERE (loan3 > 100000 AND deposit3 < loan3) AND (credit3 <= 3 OR credit3 >= 7) AND (ID > 100000 AND ID < 100030)']
# ONLINE_LIST = [0,1,0,1]
# # SQLs = ["SELECT ss_sold_date_sk FROM S_user3_table ORDER BY ss_store_sk DESC"]
# SQLs = ["SELECT ss_sold_date_sk FROM S_user3_table WHERE ss_store_sk <= 30000"]
# # SQLs = ["SELECT ID FROM S_user3_table0 ORDER BY loan3 DESC",
# #         "SELECT ID FROM S_user3_table0 WHERE (loan3 > 100000 AND (deposit3 < loan3)) AND (credit3 <= 3 OR credit3 >= 7)",
# #         "SELECT ID FROM S_user3_table0 WHERE deposit3 < 5000000 AND deposit3 < loan3",
# #         "SELECT ID FROM S_user2_table0 WHERE credit2 >= 5",
# #         "SELECT ID FROM S_user1_table0 WHERE loan1 < 80000 OR loan1 >= 150000",
# #         "SELECT ID FROM S_user0_table0 WHERE AGE < 40",
# #         "SELECT ID FROM S_user3_table0 WHERE deposit3 < loan3",
# #         "SELECT ID FROM S_user3_table0 WHERE credit3 < 6 AND deposit3 < loan3",
# #         "SELECT ID FROM S_user3_table0 WHERE deposit3 < loan3 OR (ID > 100010)",
# #         "SELECT ID FROM S_user3_table0 WHERE (loan3 >= 450000) AND ((deposit3 < loan3) OR (deposit3 <= 30000))",
# #         "SELECT ID FROM S_user3_table0 WHERE (deposit3 < loan3) AND (loan3 >= 100000)",
# #         "SELECT ID FROM S_user1_table0 WHERE credit1 >= 5",
# #         "SELECT ID FROM S_user3_table0 WHERE (deposit3 < loan3) AND (credit3 < 7)",
# #         "SELECT ID FROM S_user0_table0 WHERE AGE > 60",
# #         "SELECT ID FROM S_user2_table0 WHERE deposit2 >= loan2",
# #         "SELECT ID FROM S_user2_table0 WHERE (deposit2 >= loan2) AND (credit > 4)",
# #         "SELECT ID FROM S_user2_table0 WHERE deposit2 >= loan2 OR ID < 100500",
# #         "SELECT ID FROM S_user2_table0 WHERE (credit < 3 OR credit > 8) AND (deposit2 >= loan2)",
# #         "SELECT ID FROM S_user2_table0 WHERE deposit2 >= loan2 OR deposit2 < 300000",
# #         "SELECT ID FROM S_user3_table0 WHERE deposit3 < loan3 OR loan3 >= 2500000",
# #         "SELECT ID FROM S_user3_table0 WHERE ID < 100015 AND deposit3 < loan3",
# #         "SELECT ID FROM S_user3_table0 ORDER BY loan3 DESC"]

# p0 = multiprocessing.Process(target = executeSecurePlan, args = (0, ONLINE_LIST, SQLs))
# p1 = multiprocessing.Process(target = executeSecurePlan, args = (1, ONLINE_LIST, SQLs))
# p2 = multiprocessing.Process(target = executeSecurePlan, args = (2, ONLINE_LIST, SQLs))

# p0.daemon = True
# p0.start()
# p1.daemon = True
# p1.start()
# p2.daemon = True
# p2.start()

# p0.join()
# p1.join()
# p2.join()

# if __name__ == '__main__':
#     SQL = "SELECT deposit3,credit3 FROM S_user3_table0 WHERE deposit3>200000 AND credit<=6"
#     plan = parseSQL(SQL)
#     executeSecurePlan([0,1,0,1], plan)
