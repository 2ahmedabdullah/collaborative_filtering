#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 12:52:17 2022

@author: abdul
"""

import numpy
from matplotlib import pyplot as plt

def matrix_factorization(R, P, Q, K, steps=50000, alpha=0.0002, beta=0.02):
    '''
    R: rating matrix
    P: |U| * K (User features matrix)
    Q: |D| * K (Item features matrix)
    K: latent features
    steps: iterations
    alpha: learning rate
    beta: regularization parameter'''
    Q = Q.T
    loss = []
    for step in range(steps):
        
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # calculate error
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])

                    for k in range(K):
                        # calculate gradient with a and beta parameter
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
                        
        eR = numpy.dot(P,Q)
        e = 0

        for i in range(len(R)):

            for j in range(len(R[i])):

                if R[i][j] > 0:

                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)

                    for k in range(K):

                        e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
        loss.append(e)
        # 0.001: local minimum
        if e < 0.001:
            print('BROKE')
            break
    return P, Q.T, loss


R = [[5,3,0,1],
     [4,0,0,1],
     [1,1,0,5],
     [1,0,0,4],
     [0,1,5,4],
     [2,1,3,0]]

R = numpy.array(R)
# N: num of User
N = len(R)
# M: num of Movie
M = len(R[0])
# Num of Features
K = 3

 
P = numpy.random.rand(N,K)
Q = numpy.random.rand(M,K)

 

nP, nQ, loss = matrix_factorization(R, P, Q, K)
plt.plot(loss)

nR = numpy.dot(nP, nQ.T)

#error
unmasked = numpy.where(R != 0)

error = 0
for i in range(0, len(unmasked[0])):
    error = error + (R[unmasked[0][i]][unmasked[1][i]] - nR[unmasked[0][i]][unmasked[1][i]])
    #print(error)

avg_error = error/len(unmasked[0])
print('AVG ERROR:', round(avg_error,3))



#recommendation
masked = numpy.where(R == 0)
r_array = numpy.zeros(shape=(N, M))

for i in range(0, len(masked[0])):
    r_array[masked[0][i]][masked[1][i]] = nR[masked[0][i]][masked[1][i]]
    #print(error)

r_array[3][3] = 4

recommended = numpy.where(r_array > 3)
import pandas as pd


dic = {}
for i in range(0, N):
    k = pd.Series(r_array[i,:])
    k = k[k!=0]
    k1 = k.sort_values(ascending=False)
    k2= list(k1.index)

    dic[i] = k2


#output
dic[int(input('Enter user Id:', ))]
















