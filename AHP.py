# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 20:08:20 2020
层次分析法 实践
@author: Administrator
"""

import numpy as np
 
"""
1. 成对比较矩阵 
"""
def comparision(W0):  # W为每个信息值的权重
    n=len(W0)
    F=np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if i==j:
                F[i,j]=1
            else:
                F[i,j]=W0[i]/W0[j]
    return F


"""
2. 单层排序,相对重要度
"""
def ReImpo(F):
    n=np.shape(F)[0]
    W=np.zeros([1,n])
    for i in range(n):
        t=1
        for j in range(n):
            t=F[i,j]*t
        W[0,i]=t**(1/n)
    W=W/sum(W[0,:])  # 归一化 W=[0.874,2.467,0.464]
    return W.T

"""
3. 一致性检验
"""
def isConsist(F):
    n=np.shape(F)[0]
    a,b=np.linalg.eig(F)
    maxlam=a[0].real
    CI=(maxlam-n)/(n-1)
    if CI<0.1:
        return bool(1)
    else:
        return bool(0)


if __name__ == '__main__':
    w0 = [5,4,5,3,1]
    F = np.array([[1,3,5,7,9]])
    F = comparision(w0)
    w = ReImpo(F)
    isConsist(F)