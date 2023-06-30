# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 09:57:58 2022

@author: hanye
"""
from Compiler.types import *
from Compiler.library import *
from Compiler import ml
from Compiler.util import is_zero, tree_reduce
from Compiler import util

#1.随机产生整数向量，打包向量和其索引成为一个节点，压入list，返回一个类似于vector的结构
#2.对节点的茫然交换做定义
#3.双调排序
#4.求取topK

sfix.set_precision(16,57)
print_float_precision(20)

class secure_topK_element():
    
    def __init__(self):
        pass
    
    def Gen_Random_NodeList(self,nvals,value_type):#nvals:要生成的list的长度
        Random_NodeList= Matrix(nvals,2,value_type)#n行，2列，第一列存index，第二列存value->一行是一个node
        @for_range(nvals)
        def _(i):
            currentVal=value_type.get_random(lower=-10,upper=10)
            currentIndex=sint(i)
            Random_NodeList[i][0]=currentIndex
            Random_NodeList[i][1]=currentVal
        #     print_ln('currentNode:(%s,%s)', currentIndex.reveal(), currentVal.reveal())
        
       # @for_range(nvals)
        #def _(i):
             #print_ln('checkNode:(%s,%s)', Random_NodeList[i][0].reveal(), Random_NodeList[i][1].reveal())
        
        return Random_NodeList
    
    
    def cond_swap_with_bit(self, b, x, y):
        bx = b * x
        by = b * y
        return bx + y - by, x - bx + by
    
    # def cond_swap(self, x, y,type_sort):
    #     if(type_sort==0):
    #         b = x.__gt__(y) #私有比较
    #     elif(type_sort==1):
    #         b = x.__lt__(y)
    #     x_new, y_new = self.cond_swap_with_bit(b, x, y)
    #     return b, x_new, y_new
    

    def cond_swap_forNode(self,InputVector,Dataflag,index1,index2,type_sort):#从交换的结果来说，type=0时大->小，type=1时小->大
        x_index=InputVector[index1][0]
        y_index=InputVector[index2][0]
        
        x_value=InputVector[index1][1]
        y_value=InputVector[index2][1]

        x_flag=Dataflag[index1]
        y_flag=Dataflag[index2]

        #交换
        @if_e((x_flag == regint(0)) & (y_flag != regint(0)))
        def _():
            InputVector[index1][1] = y_value
            InputVector[index1][0] = y_index

            InputVector[index2][1] = x_value
            InputVector[index2][0] = x_index

            Dataflag[index1] = y_flag
            Dataflag[index2] = x_flag
        @else_
        def _():
            #不交换
            @if_e(Dataflag[index2] == 0)
            def _():
                pass  
            #私有比较 
            @else_
            def _():
                if(type_sort==0):
                    b = x_value.__gt__(y_value) #私有比较
                elif(type_sort==1):
                    b = x_value.__lt__(y_value)
                x_value_new, y_value_new = self.cond_swap_with_bit(b, x_value,y_value)
                x_index_new, y_index_new = self.cond_swap_with_bit(b, x_index,y_index)

                InputVector[index1][0]=x_index_new
                InputVector[index2][0]=y_index_new
                
                InputVector[index1][1]=x_value_new
                InputVector[index2][1]=y_value_new

              

    def get_secure_topK(self,DataVector,K,theta,type_sort,value_type):
        #start_timer(2)
        
        Data_row=DataVector.sizes[0]
        DataHead = Matrix(K,2,value_type)
        assert(Data_row >= K)
        if(Data_row==K or K*(1+theta)>=Data_row): #代表和一次整体排序等价
            DataSort = self.odd_even_merge_sort(DataVector, type_sort)
            @for_range_opt(K)
            def _(j):
                DataHead[j]=DataSort[j]
        else: 
            #DataHead=Matrix(K,2,value_type)
            #DataRest=Matrix(Data_row-K,2,value_type)
            #DataMask=Matrix(theta*K,2,value_type)
            every_sort_length = K*(1+theta)
            Data_tobeSorted=Matrix(every_sort_length,2,value_type)

            @for_range_opt(every_sort_length)
            def _(j):
                  Data_tobeSorted[j] = DataVector[j]

            #余下元素的首节点的 索引
            global DataRest_head
            DataRest_head =  regint(every_sort_length)
            #global lastlen
            #lastlen = regint((Data_row-K)%(theta * K))
            #DataRest_tail = Data_row-1


            DataSort = self.odd_even_merge_sort(Data_tobeSorted, type_sort)
            @for_range_opt(K)
            def _(j):
                DataHead[j]=DataSort[j]

            #print_ln("data_head:%s",DataHead.reveal_nested())

            flag_continue=MemValue(1)
            @do_while
            def _():
            #while(flag_continue==1):
                last_element = (Data_row-DataRest_head)
                #print_ln('last_element:%s',regint(last_element))
                @if_e(last_element <= theta * K)
                def _():              
                    #assert(lastlen==(Data_row-DataRest_head))
                    #print_ln('here last batch:')
                    lastlen = int((Data_row-K)%(theta * K) + K)
                    #print_ln("lastlen:%s",lastlen)

                    Data_tobeSorted=Matrix(lastlen,2,value_type)

                    @for_range_opt(len(Data_tobeSorted))
                    def _(j):
                        @if_e(j < K)
                        def _():
                            Data_tobeSorted[j] = DataHead[j]
                        @else_
                        def _():
                            global DataRest_head
                            #print_ln('DataRest_head:%s',regint(DataRest_head))
                            Data_tobeSorted[j] = DataVector[DataRest_head]
                            DataRest_head = DataRest_head+1
                    flag_continue.write(0)
                              
                @else_
                def _():
                    #Data_tobeSorted.assign(DataHead)
                    @for_range_opt(len(Data_tobeSorted))
                    def _(j):
                        @if_e(j < K)
                        def _():
                            Data_tobeSorted[j] = DataHead[j]
                        @else_
                        def _():
                            global DataRest_head
                            #print_ln('DataRest_head:%s',regint(DataRest_head))
                            Data_tobeSorted[j] = DataVector[DataRest_head]
                            DataRest_head = DataRest_head+1
                            #assert(DataRest_head <= Data_row)
                    
                DataSort = self.odd_even_merge_sort(Data_tobeSorted, type_sort)
                
                @for_range_opt(K)
                def _(j):
                    DataHead[j]=DataSort[j]

                #print_ln("data_head:%s",DataHead.reveal_nested())

                #print('flag_continue:',flag_continue)
                return_flag = regint(flag_continue.read())
                return return_flag
           
        #stop_timer(2)
        #返回索引值，sint格式
        topk_index_arr = Array(K,sint)
        topk_index_arr.assign_all(-1)
        #返回数值，sfix格式
        topk_value_arr = Array(K,sfix)
        topk_value_arr.assign_all(-1)
        @for_range_opt(K)
        def _(i):
            topk_index_arr[i] = DataHead[i][0]
            topk_value_arr[i] = DataHead[i][1]
        return topk_index_arr,topk_value_arr
        
    
    
    # modified from loopy_odd_even_merge_sort in library.py of MP-SPDZ
    def odd_even_merge_sort(self, InputVector, type_sort, sorted_length=1, n_parallel=1):
        """ Pads to power of 2, sorts, removes padding """
        length = len(InputVector)
        length_should = len(InputVector)
        while length_should & (length_should-1) != 0:
            length_should = length_should + 1

        #第1列：index，第二列：value
        DataSort = Matrix(length_should,2,sfix) 
        Dataflag = Array(length_should,regint)
        @for_range_opt(length_should)
        def _(i):
            @if_e(i < length)
            def _():
                DataSort[i][0] = InputVector[i][0]
                DataSort[i][1] = InputVector[i][1]
                Dataflag[i] = 1
            @else_
            def _():
                DataSort[i][0] = sint(i)
                DataSort[i][1] = sint(0)
                Dataflag[i] = 0

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
    
                @for_range_parallel(n_parallel // n_innermost // n_inner, n_outer)
                def loop(i):
                    @for_range_parallel(n_parallel // n_innermost, n_inner)
                    def inner(j):
                        base = i * l + j
                        step = l // k
                        if k == 2:
                            index1=base
                            index2=base + step
                            self.cond_swap_forNode(DataSort,Dataflag,index1,index2,type_sort)
                            
                        else:
                            @for_range_parallel(n_parallel, n_innermost)
                            def f(i_inner):
                                m1 = step + i_inner * 2 * step
                                m2 = m1 + base
                                index1=m2
                                index2=m2 + step
                                self.cond_swap_forNode(DataSort,Dataflag,index1,index2,type_sort)

        return DataSort
