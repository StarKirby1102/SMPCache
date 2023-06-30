#!/usr/bin/env python3

import re
import sqlparse

def parseSQL(SQL):
    '''
    Preprocessing the SQL in order to match the operator using re
    
    Operator   ReplaceCode
    >          op0
    <          op1
    ==         op2
    >=         op3
    <=         op4
    <>         op5
    '''
    SQL = SQL.replace('>=','op3')
    SQL = SQL.replace('<=','op4')
    SQL = SQL.replace('<>','op5')
    SQL = SQL.replace('>','op0')
    SQL = SQL.replace('<','op1')
    SQL = SQL.replace('==','op2')
    '''
    Operator   ReplaceCode
    +          op6
    -          op7
    *          op8
    /          op9
    '''
    SQL = SQL.replace('+','op6')
    SQL = SQL.replace('-','op7')
    SQL = SQL.replace('*','op8')
    SQL = SQL.replace('/','op9')
    # print(SQL)
    '''
    Step1. Split the SQL into a list
    This module split the SQL statement into key-value
    For example: "SELECT id FROM tableA WHERE loan>100000" will parse and split as:
        [["SELECT": id],
         ["FROM": tableA],
         ["WHERE": loan>100000]]
    '''
    parsed = sqlparse.parse(SQL)
    stmt = parsed[0]
    parsed_list = []
    for token in stmt.tokens:
        if token.value != ' ':
            if token.value.upper().__contains__("WHERE"):
                parsed_list.append("WHERE")
                parsed_list.append(token.value[6:])
            else:
                parsed_list.append(token.value)
    # print("Successfully execute step1:")
    # print(parsed_list)
    
    '''
    Step2. Combine each part to key-value

    Note: this module just for test, only support simple SELECT SQL
    '''
    SELECT_KEYWORDS = ['SELECT', 'FROM', 'WHERE', 'ORDER BY']
    INSERT_KEYWORDS = []
    UPDATE_KEYWORDS = []
    DELETE_KEYWORDS = []

    SQL_dict = {}
    for index in range(len(parsed_list)):
        if parsed_list[index].upper() in SELECT_KEYWORDS:
            SQL_dict[parsed_list[index].upper()] = parsed_list[index + 1]
            index += 1
    # print("Successfully execute step2:")
    # print(SQL_dict)

    '''
    Step3. substract the token.value into operator buffer

    For example: if the select condition is "loan>50000 AND loan<1000000", this condition will parse as:
        ['loan>50000', 'AND', 'loan<1000000']
    
    Above are three subconditions, and we'll substract the subconditions as:
        [['loan', '>', '50000'], ['AND'], ['loan', '<', '1000000']]
    
    Note: this module just for test, only support single table SELECT SQL
    '''
    
    pattern_conj = r'(\s+AND\s+|\s+OR\s+|\s*,\s*|\s*\(\s*|\s*\)\s*)'
    pattern_op = r'(\s*op1\s*|\s*op2\s*|\s*op3\s*|\s*op4\s*|\s*op5\s*|\s*op6\s*|\s*op7\s*|\s*op8\s*|\s*op9\s*|\s*op0\s*|)'
    for key in SQL_dict:
        SQL_dict[key] = re.split(pattern_conj, SQL_dict[key])
        value = []
        for tuple in SQL_dict[key]:
            tuple = re.split(pattern_op, tuple)
            if tuple != ['']:
                value.append(tuple)
        SQL_dict[key] = value

    # print("Successfully execute step3:")
    # for key in SQL_dict:
        # print("keywords: " + key)
        # print(SQL_dict[key])

    return SQL_dict


if __name__ == '__main__':
    res = parseSQL("SELECT ID,AVG(DEPOSIT),MAX(LOAN1) FROM TABLE")
    print(res)