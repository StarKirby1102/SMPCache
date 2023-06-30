#!/usr/bin/env python3

from queryParse import *
import os
import numpy as mp
import sys
import tensorflow as tf

'''
This modult aimed to transfer the infix expression to AST

For example:
Given a [WHERE] subsequence: 
'''
class TreeNode:
    '''
    Define the AST's tree node, each node has three parts:

    1. content: the node's content, e.g. 'deposit', 'op5', 50000
    2. left: left node
    3. right: right node
    '''

    '''
    Construct function
    '''
    def __init__(self, exp):
        self.content = exp
        self.left = NULL
        self.right = NULL

def priority(a):
    '''
    This function will return the priority of each operator, if priority greater, the return value will bigger
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
    if a in ['ALL', 'ANY', 'BETWEEN', 'LIKE', 'IN', 'OR', 'SOME']:
        return 1
    if a in ['AND']:
        return 2
    if a in ['NOT']:
        return 3
    if a in ['op0', 'op1', 'op2', 'op3', 'op4', 'op5']:
        return 4
    if a in ['op6', 'op7']:
        return 5
    if a in ['op8', 'op9']:
        return 6

def infix2postfix(exp):
    '''
    This function will transfer the infix expression to postfix expression

    For example: 
    ['deposit', 'op5', '20000', 'AND', '(', 'credit', 'op3', '3', 'OR', 'credit', 'op4', '7', ')']
    |   |   |   |   |
    After transform:
    ['deposit', '20000', 'credit', '3', 'op3', 'credit', '7', 'op4', 'op5', 'AND']
    '''

    '''
    tuple: Store the infix form for WHERE subquery
    '''
    op = ['ALL', 'ANY', 'BETWEEN', 'LIKE', 'IN', 'OR', 'SOME', 'AND', 'NOT', 'op0', 'op1', 'op2', 'op3', 'op4', 'op5', 'op6', 'op7', 'op8', 'op9', '(', ')']
    tuples = []
    for subexp in exp:
        for element in subexp:
            if element != '':
                tuples.append(element.replace(' ', ''))
    
    '''
    Define two stacks: stack, postfix

    stack-store the operator
    postfix-store the postfix
    '''
    stack = []
    postfix = []
    for tuple in tuples:
        if tuple not in op:
            '''
            If tuple not in operator, push to the stack
            '''
            postfix.append(tuple)
        else:
            if tuple != ')' and (not stack or tuple == '(' or stack[-1] == '('
                             or priority(tuple) > priority(stack[-1])):
                '''
                If stack not empty, stack's top is '('
                '''
                stack.append(tuple)
            elif tuple == ')':
                '''
                If tuple is ')', push into the stack
                '''
                while True:
                    temp = stack.pop()
                    if temp != '(':
                        postfix.append(temp)
                    else:
                        break
            else:
                '''
                Compare the priority of operators to decide whether to push or pop the stack
                '''
                while True:
                    if stack and stack[-1] != '(' and priority(tuple) <= priority(stack[-1]):
                        postfix.append(stack.pop())
                    else:
                        stack.append(tuple)
                        break
    
    while stack:
        '''
        If stack is not empty, all remaining elements are pushed into the postfix stack
        '''
        postfix.append(stack.pop())
    return postfix
    

def AST():
    '''
    This function will build the expression AST according to postfix expression
    '''
    node = TreeNode("")