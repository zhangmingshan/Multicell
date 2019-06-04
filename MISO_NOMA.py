# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:18:32 2017

@author: lsalaun
"""

import numpy as np
import numpy.linalg as la
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
import copy
from scipy.optimize import linear_sum_assignment

# from cvxopt import matrix, solvers
# import cvxopt

# N : Number of transmitter antenna
# K : Number of users per cell
# M : Number of cells
# H : link gain matrix, dim = [M][K][N]
# G : Link gain matrix of inter-cell link gain,, dim = [M][K][N]
# W_MF : MF beamforming matrix, dim = [M][K][N] (Matched filter)
# W_ZF : ZF beamforming matrix
# W_MMSE : MMSE beamforming matrix

# return the norm 1 (complex modulus) of the last dimension of Matrix
# The last dimension corresponds to the N antennas part
test = False

def vectNorm(M,K,N,Matrix):
    res = np.zeros((M,K,N))
    for m in range(M):
        for k in range(K):
            norm_antennas = la.norm(Matrix[m][k],ord=2)
            for n in range(N):
                res[m][k][n] = norm_antennas
    return res

def vectNorm1(K,N,Matrix):
    res = np.zeros((N,K))
    for k in range(K):
        norm_antennas = la.norm(Matrix[:,k],ord=2)
        for n in range(N):
            res[n,k] = norm_antennas
    return res

def vectNorm2(K,N,Matrix):
    res = np.zeros((N,K))
    for n in range(N):
        norm_antennas = la.norm(Matrix[n,:],ord=2)
        for k in range(K):
            res[n][k] = norm_antennas
    return res

def vectNorm3(K,N,Matrix):
    res = np.zeros((K,N))
    res = la.norm(Matrix,ord=2)
    return res
# MF beamforming matrix (normalized)
def w_mf(M,K,N,H,P = None,sigma_square = None):
#    print(H/vectNorm(M,K,N,H))
    #print('---- W_MF TEST\n',H,'\n',vectNorm(M,K,N,H),'\n',H/vectNorm(M,K,N,H),'\n-------------------------\n')
    return H/vectNorm(M,K,N,H)

# ZF beamforming matrix (normalized)
def w_zf(M,K,N,H,Sz,P = None,sigma_square = None):
    # print('Sz: ', Sz)
    W_ZF = np.zeros((M,N,N), dtype=complex)
    H_BF = np.zeros((M,N,N),dtype = complex)
    for m in range(M):
        for n in range(N):
            k_BF = int(Sz[m,n])-1
            H_BF[m,n,:] = H[m,m,k_BF,:]
            # H_BF[m,n,:] = H[m,m,k_BF,:]
        # A = H_BF[m,:,:].conj()
        A = H_BF[m,:,:].conj()
        # W_ZF[m,:,:] = np.dot(A.conj().T,la.pinv(np.dot(A,A.conj().T))).T
        W_ZF[m,:,:] = np.dot(A.conj().T,la.pinv(np.dot(A,A.conj().T)))
        # print('H: ',H[m,m,:,:])
        # print('w: ',W_ZF[m,:,:])
        # print('result: ',(abs(np.dot(H[m,m,:,:].conj(),W_ZF[m,:,:]))**2))
        # print('norm: ',vectNorm3(N,N,W_ZF[m,:,:]))
        W_ZF[m,:,:] = W_ZF[m,:,:]/vectNorm1(N,N,W_ZF[m,:,:])
        # print('result: ',(abs(np.dot(H[m,m,:,:].conj(),W_ZF[m,:,:]))**2))
        # print('w: ',W_ZF[m,:,:])
        # print('H: ',H[m,m,:,:],'; w: ',W_ZF[m,:,:],'result: ',(abs(np.dot(H[m,m,:,:].conj(),W_ZF[m,:,:]))**2))
        # print('norm: ',vectNorm3(N,N,W_ZF[m,:,:]))
    # print('norm1: ',vectNorm(M,N,N,W_ZF))
    return W_ZF # /vectNorm(M,N,N,W_ZF)

# MMSE beamforming matrix (normalized)
# P : power vector, dim [M][K]
def w_mmse(M,K,N,H,P,sigma_square):
    W_MMSE = np.zeros((M,K,N), dtype=complex)

    for m in range(M):
        A = H[m,:,:].conj()
        S = np.array([[sigma_square[m,0],0],[0,sigma_square[m,1]]], dtype=complex)
        part1 = A.conj().T
        part2 = la.inv( np.dot(A,A.conj().T) + S )
        W_MMSE[m,:,:] =  np.dot(part1,part2).T  
          
    return W_MMSE/vectNorm(M,K,N,W_MMSE)

# ================================================= NOMA (with SIC) =================================================

# Compute the inter-cell interference plus noise of user i in cell m
def I(M,K,N,H,w,S,q,m,n,k):
    # Compute result
    I = 0
    # first part
    for j in range(N):
        if j != n:
            # print('first q: ',q[m,j],' first hw: ',(abs(np.dot(H[m,m,k,:].conj(),w[m,:,j]))**2))
            # I += q[m,j]*(abs(np.dot(H[m,m,k,:].conj(),w[m,j,:]))**2)
            I += q[m,j]*(abs(np.dot(H[m,m,k,:].conj(),w[m,:,j]))**2)
    # print('first I: ',I)
    # second part
    for m_prime in range(M):
        if m_prime != m:
            for l in range(N):
                # print('inter-cell interference: ',q[m_prime,l]*(abs(np.dot(H[m,m_prime,k,:].conj(),w[m_prime,:,l]))**2))
                # I += q[m_prime,l]*(abs(np.dot(H[m,m_prime,k,:].conj(),w[m_prime,l,:]))**2)
                I += q[m_prime,l]*(abs(np.dot(H[m,m_prime,k,:].conj(),w[m_prime,:,l]))**2)
    # print('second I: ',I)
    # noise
    I += S[m][k]   
    # Normalization
    # print('I: ',I,'; normalized I: ',I/(abs(np.dot(H[m,m,k,:].conj(),w[m,:,n]))**2))
    # print('I: ',I,'; H: ',H[m,m,k,:],'; w: ',w[m,n,:],'; result: ',(abs(np.dot(H[m,m,k,:].conj(),w[m,n,:]))**2))
    # print('I: ',I,'; H: ',H[m,m,k,:],'; w: ',w[m,:,n],'; result: ',(abs(np.dot(H[m,m,k,:].conj(),w[m,:,n]))**2))
    # print('I: ',I,' result: ',(abs(np.dot(H[m,m,k,:].conj(),w[m,:,n]))**2),' normalized I: ',I/(abs(np.dot(H[m,m,k,:].conj(),w[m,:,n]))**2))
    # I = I/(abs(np.dot(H[m,m,k,:].conj(),w[m,:,n]))**2)
        
        # print('result: ',(abs(np.dot(H[m,m,:,:].conj(),w[m,:,:]))**2))
    I = I/(abs(np.dot(H[m,m,k,:].conj(),w[m,:,n]))**2)
    # I = I/(abs(np.dot(H[m,m,:,:].conj(),w[m,:,:]))**2)[k,n]
    # if I_norm == float("inf"):
        # I_norm = 1e46
        # print('I: ',I,' result: ', (abs(np.dot(H[m,m,k,:].conj(),w[m,:,n]))**2))
    # print('I: ',I)
    # I = I/(abs(np.dot(H[m,m,k,:].conj(),w[m,:,n]))**2)
    return I

# Given a set of constraints Gamma and decoding order pi, 
# and the power vector p compute the next power vector p_next according to f
# Gamma : SINR constraints, dim [M][K]
# Gamma[m][0] constraints of the first user in pi[m] (no SIC)
# Gamma[m][1] constraints of the second user in pi[m] (with SIC)
# m : current cell
# p : current power vector, dim [M][K]
# pi : decoding order in cell m, dim [K]. 
# pi[0] index of first decoded user (with pi[1] signal as interference) in cell m
# pi[1] index of second decoded user (no interference because of SIC) in cell m
# W : beamforming vector generator lambda function
# H and G
# S : array of sigma's, dim [M][K] 
# p_next [output] : next power vector in cell m, dim [K]
def f(M,K,N,H,w,S,Gamma,pi,q_prev,p_prev,m):
    # print('*************f***********')
    # print('pi: ',pi)
    # Compute p_next
    q_next = np.zeros((N))
    p_next = np.zeros((N,K))
    
    # Q-OPT
    # print('Q-OPT')
    X = np.zeros((K))
    for n in range(N):
        L = len(pi[n,:].ravel()[np.flatnonzero(pi[n,:])])
        # print('L: ',L)
        for i in range(L):
            k = pi[n,:].tolist().index(i+1)
            X[k] = Gamma[m,k]
            for i_prime in range(i):
                k_prime = pi[n,:].tolist().index(i_prime+1)
                X[k] = X[k]*(Gamma[m,k_prime]+1)

        for i in range(L):
            k = pi[n,:].tolist().index(i+1)
            # if I(M,K,N,H,w,S,q_prev,m,n,k) == float("inf"):
                # print('pi: ',pi)
                # print('m: ',m,'n:',n,' k:',k)
            q_next[n] += X[k]*I(M,K,N,H,w,S,q_prev,m,n,k)
    
    # P_OPT
    # print('P-OPT')
    for n in range(N):
        L = len(pi[n,:].ravel()[np.flatnonzero(pi[n,:])])
        # print('L: ',L)
        k_L = pi[n,:].tolist().index(L)
        p_next[n,k_L] = Gamma[m,k_L]*I(M,K,N,H,w,S,q_prev,m,n,k_L)
        for i in range(L-1,0,-1):
            p_temp = 0
            for i_prime in range(i,L): # L-1):
                k_i_prime = pi[n,:].tolist().index(i_prime+1)
                p_temp += p_next[n,k_i_prime]
            k_i = pi[n,:].tolist().index(i)
            p_next[n,k_i] = Gamma[m,k_i]*(p_temp+I(M,K,N,H,w,S,q_prev,m,n,k_i))

    # print('q: ',q_next)
    return q_next, p_next


def SUS(m,K,N,H,alpha):
    # Step I: Initialization
    T = np.arange(1,K+1,1)
    # print('T:, ',T)
    S0 = np.zeros((N))
    h = np.zeros((N,N), dtype=complex)
    g = np.zeros((N,N), dtype=complex)
    i = 0
    # buffers of gk
    gk = np.zeros((K,N), dtype=complex)
    while(S0[N-1]==0 and i<K):
        # Step2: for each user k in T, calculate gk
        if i==0:
            for k in T:
                gk[k-1,:]=H[m,m,k-1,:]
        else:
            for k in T:
                if k==0:
                    pass
                else:
                    temp = 0
                    for j in range(i):
                        temp = temp+(np.dot(g[j,:].T,g[j,:])/la.norm(g[j,:],1)**2)
                    gk[k-1,:] = np.dot(H[m,m,k-1,:],(np.eye(N)-temp))
        # Step III: Select the i-th BF user
        Idx = False
        for k in T:
            if T[k-1]!=0:
                Idx = False
        normG = np.zeros((K))
        for k in T:
            if (k==0):
                pass
            else:
                normG[k-1] = la.norm(gk[k-1,:],1)
        if np.max(normG) == 0:
            Idx = True

        MAXI = np.argmax(normG)
        # Randomly choose the next beamforming user if the semiorthogonal user set is empty
        while Idx:
            MAXI = random.randint(0,K-1)
            for r in range(i):
                if MAXI==S0[r]-1:
                    break
                else:
                    Idx = False
        S0[i] = MAXI+1
        PHI = MAXI
        
        h[i,:] = H[m,m,PHI,:]
        g[i,:] = gk[PHI,:]
        
        # Step IV: Update the set of users semiortoghonal to g(i)
        T[PHI] = 0
        for k in T:
            if k==0:
                pass
            else:
                if (k-1)==PHI:
                    T[k-1] = 0
                else:
                    if abs(np.dot(H[m,m,k-1,:],g[i,:]))/(la.norm(H[m,m,k-1,:],1)*la.norm(g[i,:],1))<alpha:
                        T[k-1] = k
                    else:
                        T[k-1] = 0
        i = i+1
    return S0

def clustering(M,K,N,H,W,S,Gamma,alpha,epsilon,iterations=100,init_p=None):
    # Select for BF users by SUS
    Sz = np.zeros((M,N))
    for m in range(M):
        Sz[m,:] = SUS(m,K,N,H,alpha)
    # print('Sz: ',Sz)
    # Select for matched users by channel correlations
    Szm = np.zeros((M,N,K))
    for m in range(M):
        for k in range(K):
            idx = 0
            CorrMax = 0
            for n in range(N):
                Szn = int(Sz[m,n]-1)
                if Szn == k:
                    idx = n
                    break
                Corr = abs(np.dot(H[m,m,Szn,:].conj().T,H[m,m,k,:]))/(la.norm(H[m,m,Szn,:],1)*la.norm(H[m,m,k,:],1))
                if Corr>CorrMax:
                    CorrMax = Corr
                    idx = n
            Szm[m,idx,k] = 1
    # print('Szm: ',Szm)
    return Sz,Szm
    
def decodingOrder(M,K,N,H,w,S,q_prev,Szm_mn,m,n):
    # print('*************decodingOrder*************')
    pi = np.zeros(K)
    Interference = np.zeros((K))
    for k in range(K):
        Interference[k] = I(M,K,N,H,w,S,q_prev,m,n,k)
    I_pi = sorted(Interference)
    I_pi.reverse()
    
    idx = 1
    for k in range(K):
        C = I_pi[k]
        if Szm_mn[Interference.tolist().index(C)] != 0 and pi[Interference.tolist().index(C)]==0:
        # if Szm_mn[Interference.tolist().index(C)] != 0:
            # if pi[Interference.tolist().index(C)]!=0:
                # print('I_pi: ',I_pi)
            pi[Interference.tolist().index(C)] = idx
            idx = idx+1
    # print('pi: ',pi)
    return pi

# Check if the power allocation p is feasible with respect to the SINR constraints Gamma
# SINR > Gamma - epsilon (due to algo_NOMA termination condition and approximation)
def isFeasibleNOMA(M,K,N,H,w,S,Gamma,pi,q,p,Sz,epsilon):
    # Get the beamforming vector
    # print('isFeasibleNOMA')
    # print('Sz: ',Sz)
    # w = W(M,K,N,H,Sz)
    res = True
    for m in range(M):
        for n in range(N):
            L = len(pi[m,n,:].ravel()[np.flatnonzero(pi[m,n,:])])
            k_L = pi[m,n,:].tolist().index(L)
            # if(sum(sum(q))>1):
                # print('q: ',sum(sum(q)),' k_L: ',k_L,' m: ',m,' p: ',p[m,n,k_L],' I: ',I(M,K,N,H,w,S,q,m,n,k_L))
            if p[m,n,k_L] - Gamma[m,k_L]*I(M,K,N,H,w,S,q,m,n,k_L) < -epsilon:
                # print(False)
                # return False
                res = False
                break
            for i in range(L-1,0,-1):
                p_temp = 0
                for i_prime in range(i,L-1):
                    k_i_prime = pi[m,n,:].tolist().index(i_prime+1)
                    p_temp += p[m,n,k_i_prime]
                k_i = pi[m,n,:].tolist().index(i)
                if p[m,n,k_i] - Gamma[m,k_i]*(p_temp+I(M,K,N,H,w,S,q,m,n,k_i))< -epsilon:
                    # return False
                    res = False
                    break
    # print('Feasible: ',res)
    return res

def algoNOMA_disjoint(M,K,N,H,W,S,Gamma,alpha,epsilon,iterations=100,init_p=None):
    Sz, Szm = clustering(M,K,N,H,W,S,Gamma,alpha,epsilon,iterations=100,init_p=None)
    
    pi_all = np.zeros((M,N,K), dtype=np.int)     
    # Power control algorithm
    p_prev = np.ones((M,N,K))*np.inf
    p_next = np.zeros((M,N,K))
    q_prev = np.ones((M,N))*np.inf
    q_next = np.zeros((M,N))
    if init_p is not None:
        p_next = init_p

    # Get the beamforming vector
    w = W(M,K,N,H,Sz)

    count = 0
    P = []
    while la.norm((p_next-p_prev).flatten(),1) > epsilon and count < iterations:
        count += 1

        p_prev = np.array(p_next)
        q_prev = np.array(q_next)
        for m in range(M):
            for n in range(N):
                pi_all[m,n,:] = decodingOrder(M,K,N,H,w,S,q_prev,Szm[m,n,:],m,n)
        for m in range(M):
            q_next[m,:],p_next[m,:,:] = f(M,K,N,H,w,S,Gamma,pi_all[m,:,:],q_prev,p_prev,m)
            if sum(q_next[m,:])==float("inf"):
                print('Inf power. ')
                print('Szm: ',Szm[m,:,:])
                print('pi_all: ',pi_all[m,:,:])
        P.append(list(np.sum(q_next,axis=1)))

    P = np.reshape(P, (count,M))     
    global test
    if test:
        plt.figure()
        for m in range(M):
            plt.plot(np.arange(count),P[:,m],label = 'm'+str(m)+'')
        plt.legend()
        plt.title('Interference-aware user clustering scheme convergence , N = '+str(N)+'')
        plt.xlabel('Number of iterations')
        plt.ylabel('Transmit power (W)')
        plt.grid()
        plt.show()
    
    isFeasible = isFeasibleNOMA(M,K,N,H,w,S,Gamma,pi_all,q_next,p_next,Sz,epsilon)
    return q_next, p_next, pi_all, count, isFeasible

# Find the minimum value in a 2D matrix
def findMin(Matrix):
    new_data = []
    for i in range(len(Matrix)):
        new_data.append(min(Matrix[i]))
    minValue = min(new_data)
    n = new_data.index(minValue)
    k = Matrix[n,:].tolist().index(minValue)
    return minValue, n, k

def algoNOMA_IAware(M,K,N,H,W,S,Gamma,alpha,epsilon,iterations=100,init_p=None):
    # Select for BF users by SUS
    Sz = np.zeros((M,N))
    for m in range(M):
        Sz[m,:] = SUS(m,K,N,H,alpha)
    # print('Sz: ',Sz)

    # The decoding order pi is decided by the interference
    pi_all = np.zeros((M,N,K), dtype=np.int)
           
    # Power control algorithm
    p_prev = np.ones((M,N,K))*np.inf
    p_next = np.zeros((M,N,K))
    q_prev = np.ones((M,N))*np.inf
    q_next = np.zeros((M,N))
    if init_p is not None:
        p_next = init_p

    # Get the beamforming vector
    w = W(M,K,N,H,Sz)
    
    count = 0
    P = []
    while la.norm((p_next-p_prev).flatten(),1) > epsilon and count < iterations:
        count += 1
        Szm = np.zeros((M,N,K))

        p_prev = np.array(p_next)
        q_prev = np.array(q_next)
        for m in range(M):
            Interferences = np.zeros((N,K))
            for n in range(N):
                for k in range(K):
                    if k!= Sz[m,n]-1:
                        Interferences[n,k] = I(M,K,N,H,w,S,q_prev,m,n,k)
            nRestUser = K
            L = np.zeros(N)
            for n in range(N):
                L[n] = math.floor((K-(n+1))/N)+1
            while(nRestUser>0):
                minValue,minN,minK = findMin(Interferences)
                Szm[m,minN,minK] = 1
                for n in range(N):
                    Interferences[n,minK] = np.inf
#                if sum(Szm[m,minN,:])>=L[minN]:
#                    for k in range(K):
#                        Interferences[minN,k] = np.inf
                nRestUser  = nRestUser-1
                
            # The decoding order pi is decided by the interference
            for n in range(N):
                pi_all[m,n,:] = decodingOrder(M,K,N,H,w,S,q_prev,Szm[m,n,:],m,n)
        for m in range(M):
            q_next[m,:],p_next[m,:,:] = f(M,K,N,H,w,S,Gamma,pi_all[m,:,:],q_prev,p_prev,m)

        P.append(list(np.sum(q_next,axis=1)))
    P = np.reshape(P, (count,M)) 
    
    global test
    if test:
        plt.figure()
        for m in range(M):
            plt.plot(np.arange(count),P[:,m],label = 'm'+str(m)+'')
        plt.legend()
        plt.title('Interference-aware user clustering scheme convergence , N = '+str(N)+'')
        plt.xlabel('Number of iterations')
        plt.ylabel('Transmit power (W)')
        plt.grid()
        plt.show()
    
    isFeasible = isFeasibleNOMA(M,K,N,H,w,S,Gamma,pi_all,q_next,p_next,Sz,epsilon)
    return q_next, p_next, pi_all, count, isFeasible

def algoNOMA_PAware(M,K,N,H,W,S,Gamma,alpha,epsilon,iterations=100,init_p=None):
    # Select for BF users by SUS
    Sz = np.zeros((M,N))
    for m in range(M):
        Sz[m,:] = SUS(m,K,N,H,alpha)

    # print('Sz: ',Sz)
    # The decoding order pi is decided by the interference
    pi_all = np.zeros((M,N,K), dtype=np.int)
           
    # Power control algorithm
    p_prev = np.ones((M,N,K))*np.inf
    p_next = np.zeros((M,N,K))
    q_prev = np.ones((M,N))*np.inf
    q_next = np.zeros((M,N))
    if init_p is not None:
        p_next = init_p
        
    count = 0
    # Get the beamforming vector
    w = W(M,K,N,H,Sz)

    P = []
    while la.norm((p_next-p_prev).flatten(),1) > epsilon and count < iterations:
        count += 1
        Szm = np.zeros((M,N,K))
        # The beamforming users are put into their cluster
        for m in range(M):
            for n in range(N):
                Szm[m,n,int(Sz[m,n])-1] = 1
        

        p_prev = np.array(p_next)
        q_prev = np.array(q_next)
        for m in range(M):
            Pm_prev = 0
            pi_temp = np.zeros((N,K))
            # Szm_temp = Szm[m,:,:]
            Szm_temp = copy.deepcopy(Szm[m,:,:])
            nRestUser = K-N
            Pm_next = np.zeros((N,K))
            for j in range(N):
                for n in range(N):
                    Pm_next[n,int(Sz[m,j])-1] = np.inf
            while(nRestUser>0):
                for k in range(K):
                    if sum(Szm[m,:,k]) == 0:
                        for n in range(N):
                            # print('k: ',k,'n: ',n)
                            Szm_temp = copy.deepcopy(Szm[m,:,:])
                            Szm_temp[n,k] = 1
                            for n_temp in range(N):
                                pi_temp[n_temp,:] = decodingOrder(M,K,N,H,w,S,q_prev,Szm_temp[n_temp,:],m,n_temp)
                            # print('???????????????')
                            q_temp,p_temp = f(M,K,N,H,w,S,Gamma,pi_temp,q_prev,p_prev,m)
                            Pm_next[n,k] = sum(q_temp) # q_temp[n]
                minValue,minN,minK = findMin(Pm_next-Pm_prev)
                Pm_prev = Pm_next[minN,minK]
                for n in range(N):
                    Pm_next[n,minK] = np.inf
                Szm[m,minN,minK] = 1
                nRestUser = nRestUser-1

            # The decoding order pi is decided by the interference
            # print('Szm: ',Szm)
            for n in range(N):
                pi_all[m,n,:] = decodingOrder(M,K,N,H,w,S,q_prev,Szm[m,n,:],m,n)
        for m in range(M):
            q_next[m,:],p_next[m,:,:] = f(M,K,N,H,w,S,Gamma,pi_all[m,:,:],q_prev,p_prev,m)

        P.append(list(np.sum(q_next,axis=1)))
    P = np.reshape(P, (count,M)) 
    
    global test
    # test = True
    if test:
        plt.figure()
        for m in range(M):
            plt.plot(np.arange(count),P[:,m],label = 'm'+str(m)+'')
        plt.legend()
        plt.title('Power-aware user clustering scheme convergence , N = '+str(N)+'')
        plt.xlabel('Number of iterations')
        plt.ylabel('Transmit power (W)')
        plt.grid()
        plt.show()
    
    isFeasible = isFeasibleNOMA(M,K,N,H,w,S,Gamma,pi_all,q_next,p_next,Sz,epsilon)
    return q_next, p_next, pi_all, count, isFeasible

def algoNOMA_best(M,K,N,H,W,S,Gamma,alpha,epsilon,iterations=100,init_p=None):
    # Select for BF users by SUS
    Sz = np.zeros((M,N))
    for m in range(M):
        Sz[m,:] = SUS(m,K,N,H,alpha)
    # print('Sz: ', Sz)
    # The decoding order pi is decided by the interference
    pi_all = np.zeros((M,N,K), dtype=np.int)
    # Power control algorithm
    p_prev = np.ones((M,N,K))*np.inf
    p_next = np.zeros((M,N,K))
    q_prev = np.ones((M,N))*np.inf
    q_next = np.zeros((M,N))
    if init_p is not None:
        p_next = init_p
        
    count = 0
    # Get the beamforming vector
    w = W(M,K,N,H,Sz)
    # print('w: ',w)

    P = []
    while la.norm((p_next-p_prev).flatten(),1) > epsilon and count < iterations:
        count += 1
        # print('count: ',count)
        Szm = np.zeros((M,N,K))
        # The beamforming users are put into their cluster
        # for m in range(M):
            # for n in range(N):
                # Szm[m,n,int(Sz[m,n])-1] = 1

        p_prev = np.array(p_next)
        q_prev = np.array(q_next)
        for m in range(M):
            cost = np.zeros((K,N*K))
            for n in range(N):
                for l in range(K):
                    # X = np.zeros(K)
                    for k in range(K):
                        # print('l: ',l)
                        X = Gamma[m,k]*(1+Gamma[m,k])**l
                        # cost[n*N+l,k] = X*I(M,K,N,H,w,S,q_prev,m,n,k)
                        cost[k,n*K+l] = X*I(M,K,N,H,w,S,q_prev,m,n,k)
                        # print('cost: ',cost[n*N+l,k])
            # print('cost: ',cost)
            row_ind, col_ind = linear_sum_assignment(cost)
            # P_m = cost[row_ind, col_ind].sum()
            # print('row_ind: ',row_ind)
            # print('col_ind: ',col_ind)
            for k in range(K):
                Szm[m,int(col_ind[k]/K),int(row_ind[k])] = 1
                # pi_all[m,int(row_ind[k]/N),int(col_ind[k])] = row_ind[k]%N+1
            
            # The decoding order pi is decided by the interference
            for n in range(N):
                pi_all[m,n,:] = decodingOrder(M,K,N,H,w,S,q_prev,Szm[m,n,:],m,n)
        for m in range(M):
            q_next[m,:],p_next[m,:,:] = f(M,K,N,H,w,S,Gamma,pi_all[m,:,:],q_prev,p_prev,m)

        P.append(list(np.sum(q_next,axis=1)))
    P = np.reshape(P, (count,M)) 
    
    global test
    # test = True
    if test:
        plt.figure()
        for m in range(M):
            plt.plot(np.arange(count),P[:,m],label = 'm'+str(m)+'')
        plt.legend()
        plt.title('Power-aware user clustering scheme convergence , N = '+str(N)+'')
        plt.xlabel('Number of iterations')
        plt.ylabel('Transmit power (W)')
        plt.grid()
        plt.show()
    
    isFeasible = isFeasibleNOMA(M,K,N,H,w,S,Gamma,pi_all,q_next,p_next,Sz,epsilon)
    return q_next, p_next, pi_all, count, isFeasible

# ================================================= OMA (with K time slots) =================================================

# ZF beamforming matrix (normalized)
def w_zf_OMA(M,K,N,H):
    W_ZF = np.zeros((M,N,K), dtype=complex)
    # H_BF = np.zeros((M,K,N),dtype = complex)
    # W_ZF = np.zeros((M,N,K), dtype=complex)
    H_BF = np.zeros((M,K,N),dtype = complex)
    for m in range(M):
        for k in range(K):
            H_BF[m,k,:] = H[m,m,k,:]
            # print('H_BF: ',H_BF[m,n,:])
            # print('H: ',H[m,m,k_BF-1,:])
        A = H_BF[m,:,:].conj()
        # A = H_BF[m,:,:]
        # print('A: ',A)
        # print('A: ',A)
        # print('result: ',np.dot(A,A.conj().T))
        W_ZF[m,:,:] = np.dot(A.conj().T,la.pinv(np.dot(A,A.conj().T)))
        # W_ZF[m,:,:] = np.dot(A.conj().T,la.pinv(np.dot(A,A.conj().T)))
        W_ZF[m,:,:] = W_ZF[m,:,:]/vectNorm1(K,N,W_ZF[m,:,:])
        # W_ZF[m,:,:] = la.pinv(A)
        # print('H*W: ',np.dot(H[m,m,:,:].conj() ,W_ZF[m,:,:]))
        
    # print('W_ZF: ',W_ZF,'norm: ', vectNorm(M,K,N,W_ZF))
    # return W_ZF/vectNorm(M,K,N,W_ZF)
    return W_ZF# /vectNorm3(M,K,N,W_ZF)

# Same as I but for OMA system 
# k-th user in all BSs shares the same bandwidth
def I_OMA(M,K,N,H,w,S,p,m,k):
    # print('w: ',w)
    # Compute result
    I = 0
    # Interference from the ith user of the other cell m2
    for m_prime in range(M):
        if m_prime != m:
            # print('result: ',abs(np.dot(H[m,m_prime,k,:].conj(),w[m_prime,k,:]))**2)
            I += p[m_prime,k]*(abs(np.dot(H[m,m_prime,k,:].conj(),w[m_prime,:,k]))**2)
            # if p[m_prime,k]<1 and I>1:
                # print('p: ',p[m_prime,k],'; H: ',H[m,m_prime,k,:],'; w:',w[m_prime,k,:],'; I:',I)
    # noise
    I += S[m][k]
    """
    if I>1e45:
        I = 1e45
    
    # I = I/(abs(np.dot(H[m,m,k,:].conj(),w[m,k,:]))**2)
    if (abs(np.dot(H[m,m,k,:].conj(),w[m,:,k]))**2)<1e-45:
        I = I/1e-45
        # print('OMA H*w==0. m: ',m, ' k: ',k)
    else:
        I = I/(abs(np.dot(H[m,m,k,:].conj(),w[m,:,k]))**2)
    """
    I = I/(abs(np.dot(H[m,m,k,:].conj(),w[m,:,k]))**2)
    # I = I/(abs(np.dot(H[m,m,:,:].conj(),w[m,:,:]))**2)[k,k]
    # if I>100:
        # print('I: ',I,'; H*w: ',(abs(np.dot(H[m,m,k,:].conj(),w[m,k,:]))**2))
    return I

# Same as f but for OMA SINR constraints
def f_OMA(M,K,N,H,w,S,Gamma,p_prev,m):
    # Compute p_next
    p_next = np.zeros(K)
    # P_OPT
    for k in range(K):
        p_next[k] = Gamma[m,k]*I_OMA(M,K,N,H,w,S,p_prev,m,k)
    return p_next

def isFeasibleOMA(M,K,N,H,w,S,Gamma,p,Sz,epsilon):
    # Get the beamforming vector
    # w = W(M,K,N,H)
    res = True
    for m in range(M):
        for k in range(K):
            if p[m,k] - Gamma[m,k]*I_OMA(M,K,N,H,w,S,p,m,k) < -epsilon:
                return False
                # res = False
                # break
    return res

# OMA : there is K time slots, user k use time slot k.
# Same king of distributed iterative power control in OMA 
# At each iteration, each BS chooses its transmit power to meet its SINR constraints
# with the lowest feasible power and assuming powers from the other BS are fixed 
def algoOMA(M,K,N,H,W,S,Gamma,epsilon,iterations=100):
    Sz = np.zeros((M,N))
    for m in range(M):
        for n in range(N):
            Sz[m,n] = n+1
    # Get the beamforming vector
    w = W(M,K,N,H)
    p_prev = np.ones((M,K))*np.inf
    p_next = np.zeros((M,K))
    count = 0
    P = []
    while la.norm((p_next-p_prev).flatten(),1) > epsilon and count < iterations:
        count += 1
        p_prev = np.array(p_next)
        for m in range(M):
            # update p_next
            p_next[m] = f_OMA(M,K,N,H,w,S,Gamma,p_prev,m)

        # print(p_next)
        P.append(list(np.sum(p_next,axis=1)))
    P = np.reshape(P, (count,M)) 
    # print('P.size: ',np.array(P).shape)
    # print('P: ',P)
    # P = np.reshape(P, (count,M)) 
    # print('P.size: ',np.array(P).shape)
    
    global test
    # test = True
    if test:
        plt.figure()
        for m in range(M):
            plt.plot(np.arange(count),P[:,m],label = 'm'+str(m)+'')
        plt.legend()
        plt.title('Power-aware user clustering scheme convergence , N = '+str(N)+'')
        plt.xlabel('Number of iterations')
        plt.ylabel('Transmit power (W)')
        plt.grid()
        plt.show()
    
    isFeasible = isFeasibleOMA(M,K,N,H,w,S,Gamma,p_next,Sz,epsilon)
    return p_next, count, isFeasible
"""
# The proposed distributed algorithm, where the best decoding order is chosen at each iteration
def algoNOMA_bestResponse(M,K,N,H,G,W,S,Gamma,epsilon,iterations=100,init_p=None):
    p_prev = np.ones((M,K))*np.inf
    p_next = np.zeros((M,K))
    if init_p is not None:
        p_next = init_p
    pi_res = np.zeros((M,K), dtype=np.int)
    count = 0
    while la.norm((p_next-p_prev).flatten(),1) > epsilon and count < iterations:
        count += 1
        p_prev = np.array(p_next)
        for m in range(M):
            p_min_m = np.ones(K)*np.inf
            pis = [[0,1],[1,0]]
            for pi in pis:
                p_tmp = f(M,K,N,H,G,W,S,Gamma,pi,p_prev,m)
                if la.norm(p_tmp,1) < la.norm(p_min_m,1) :
                    p_min_m = p_tmp
                    pi_res[m] = pi
            # update p_next
            p_next[m] = np.array(p_min_m)
    
#        print(p_next)
#        print(pi_res)
        
    return p_next, pi_res, count

# The modified algo : test the 4 possible decoding orders. For each decoding order, do the simulation
# only work for M = 2, K = 2
def algoNOMA_allPossibilities(M,K,N,H,G,W,S,Gamma,epsilon,iterations=100):
    p_prev = np.ones((4,M,K))*np.inf
    p_next = np.zeros((4,M,K))
    count = np.zeros(4)

    # Decoding order of the 4 possibilities
    pis = [[[0,1],[0,1]],[[1,0],[0,1]],[[0,1],[1,0]],[[1,0],[1,0]]]
    # For each decoding order :
    for i in range(4):
        while la.norm((p_next[i]-p_prev[i]).flatten(),1) > epsilon and count[i] < iterations:
            count[i] += 1
            p_prev[i] = np.array(p_next[i])
            for m in range(M):
                p_next[i][m] = f(M,K,N,H,G,W,S,Gamma,pis[i][m],p_prev[i],m)
        
#            print(p_next)

    pi_res = np.zeros((M,K), dtype=np.int)
    pi_res = pis[0]
    p_res = np.ones((M,K))*np.inf
#    p_res = p_next[0]
    feasible = False
    for i in range(4):
        if isFeasibleNOMA(M,K,N,H,G,w_mf,S,Gamma,pis[i],p_next[i],epsilon) and la.norm(p_next[i].flatten(),1) <= la.norm(p_res.flatten(),1):
            p_res = p_next[i]
            pi_res = pis[i]
            feasible = True
    if not feasible:
        p_res = None
    
    for i in range(4):
        print('i',i)
        print(p_next[i])
        print(np.sum(p_next[i]))
        print('isFeasible?',isFeasibleNOMA(M,K,N,H,G,w_mf,S,Gamma,pis[i],p_next[i],epsilon))
    
    return p_res, pi_res, count

# At each iteration algoNOMA_bestResponse is performed with
# proba 'proba' and fixed decoding order with proba '1-proba'
# initially : proba = 1
# at each iteration proba decreases
def algoNOMA_mix(M,K,N,H,G,W,S,Gamma,epsilon,iterations=100,alpha=0.99,init_p=None,seed=None):
    p_prev = np.ones((M,K))*np.inf
    p_next = np.zeros((M,K))
    if init_p is not None:
        p_next = init_p
    pi_res = np.array([[0,1],[0,1]], dtype=np.int)
    count = 0
    # Set initial proba
    proba = 1  
    # set randomizer's seed
    random.seed(seed)
    # stop condition
    termination = False
    while count < iterations and not termination:
        # check proba
        if random.random() < proba: # perform algoNOMA, change the decoding order to the current best decoding order
            count += 1
            p_prev = np.array(p_next)
            for m in range(M):
                p_min_m = np.ones(K)*np.inf
                pis = [[0,1],[1,0]]
                for pi in pis:
                    p_tmp = f(M,K,N,H,G,W,S,Gamma,pi,p_prev,m)
                    if la.norm(p_tmp,1) < la.norm(p_min_m,1) :
                        p_min_m = p_tmp
                        pi_res[m] = pi
                p_min_m = f(M,K,N,H,G,W,S,Gamma,pi_res[m],p_prev,m)
                # update p_next
                p_next[m] = np.array(p_min_m)

        else: # perform algoNOMA with the same decoding order as the previous iteration
            count += 1
            p_prev = np.array(p_next)
            for m in range(M):
                p_next[m] = f(M,K,N,H,G,W,S,Gamma,pi_res[m],p_prev,m)
            # check termination conditioon
#            print('TEST',la.norm((p_next-p_prev).flatten(),1))
            if la.norm((p_next-p_prev).flatten(),1) <= epsilon :
                termination = True
        
        # Decrease proba
        proba *= alpha
#        print('proba',proba)
        
    return p_next, pi_res, count

# Output all iteration's power vector
def algoNOMA_bestResponse_fullOutput(M,K,N,H,G,W,S,Gamma,epsilon,iterations=100,init_p=None):
    p_prev = np.ones((M,K))*np.inf
    p_next = np.zeros((M,K))
    if init_p is not None:
        p_next = init_p
    pi_res = np.zeros((M,K), dtype=np.int)
    count = 0
    p_list = []
    while la.norm((p_next-p_prev).flatten(),1) > epsilon and count < iterations:
        count += 1
        p_prev = np.array(p_next)
        for m in range(M):
            p_min_m = np.ones(K)*np.inf
            pis = [[0,1],[1,0]]
            for pi in pis:
                p_tmp = f(M,K,N,H,G,W,S,Gamma,pi,p_prev,m)
                if la.norm(p_tmp,1) < la.norm(p_min_m,1) :
                    p_min_m = p_tmp
                    pi_res[m] = pi
            # update p_next
            p_next[m] = np.array(p_min_m)
        p_list.append(np.array(p_next))
#        print(p_next)
#        print(pi_res)
        
    return p_list, pi_res, count


# ================================================= NOMA_noSIC (with beamforming) =================================================

# The proposed distributed algorithm
def algoNOMA_noSIC(M,K,N,H,G,W,S,Gamma,epsilon):
    # Initial p
    p = np.zeros((M,K))
    # Get the beamforming vector
    w = W(M,K,N,H,p,S)  
    
    # Linear programming optimization
    # Minimize: c^T * x
    # Subject to: A_ub * x <= b_ub
    # A_eq * x == b_eq
    c = np.ones(4)
    # Inequality constraints
    # First 4 rows are the conditions p_i >= 0 for i in [1..4]
    # Last 4 rows are the SINR requirement conditions
    # Normalized version : divises each row by S*Gamma (SG)
    A_ub = np.zeros((8,4))
    b_ub = np.zeros(8)
    for i in range(4):
        A_ub[i][i] = -1
    for m in range(M):
        for i in range(K):
            m2 = (m+1)%2
            print('m2: ',m2)
            j = (i+1)%2
            # Compute constant part b_ub = -I*Gamma
            SG = S[m][i]*Gamma[m][i]
            b_ub[4+m*2+i] = -1 # = -S*Gamma/SG
            # Compute the coeffs matrix A_ub
            # Self BS m ------------------
            # Self user i -------
            A_ub[4+m*2+i][m*2+i] = -abs(np.dot(H[m][i].conj(),w[m][i]))**2 / SG
            # Other user j ------
            A_ub[4+m*2+i][m*2+j] = Gamma[m][i]*abs(np.dot(H[m][i].conj(),w[m][j]))**2 / SG
            # Other BS m2 -----------------
            A_ub[4+m*2+i][m2*2+i] = Gamma[m][i]*abs(np.dot(G[m2][i].conj(),w[m2][i]))**2 / SG
            A_ub[4+m*2+i][m2*2+j] = Gamma[m][i]*abs(np.dot(G[m2][i].conj(),w[m2][j]))**2 / SG

    print(c)
    print(A_ub[4:8])
    print(b_ub)
    
    # Solve this linear programming problem    
    C = matrix(c)
    A = matrix(A_ub)
    B = matrix(b_ub)
    # Do not show the progress
    solvers.options['show_progress'] = False
    sol = solvers.lp(C,G=A,h=B)
    return sol

# Check if the power allocation p is feasible with respect to the SINR constraints Gamma
# SINR > Gamma - epsilon (due to algo_NOMA termination condition and approximation)
def isFeasibleNOMA_noSIC(M,K,N,H,G,W,S,Gamma,p,epsilon):
    # Get the beamforming vector
    w = W(M,K,N,H,p,S)
    i1 = 0
    i2 = 1
        
    res = True
    for m in range(M):
        SINR1 = (p[m][i1] * abs(np.dot(H[m][i1].conj(),w[m][i1]))**2) / (I(M,K,N,H,G,w,S,p,m,i1) + p[m][i2]*abs(np.dot(H[m][i1].conj(),w[m][i2]))**2)
        SINR2 = (p[m][i2] * abs(np.dot(H[m][i2].conj(),w[m][i2]))**2) / (I(M,K,N,H,G,w,S,p,m,i2) + p[m][i1]*abs(np.dot(H[m][i2].conj(),w[m][i1]))**2)
        res = res and (SINR1 - Gamma[m][0] > -epsilon) and (SINR2 - Gamma[m][1] > -epsilon)
        
#        print(SINR1)
#        print(SINR2)
   
    return res

# ================================================= OMA (with 2 time slots) =================================================

# Same as I but for OMA system 
# ith user in BS 1 and 2 shares the same bandwidth
def I_OMA(M,K,N,H,G,w,S,p,m,i):
    # Get the index of the other cell
    m2 = (m+1)%2
    # Compute result
    I = 0
    # Interference from the ith user of the other cell m2
    I += p[m2][i]*(abs(np.dot(G[m2][i].conj(),w[m2][i]))**2)
    # noise
    I += S[m][i]
    
    return I

# Same as f but for OMA SINR constraints
def f_OMA(M,K,N,H,G,w,S,Gamma,p,m):
    # Compute p_next
    p_next = np.zeros(K)
    
    i1 = 0 # First user
    i2 = 1 # Second user

    p_next[i1] = I_OMA(M,K,N,H,G,w,S,p,m,i1) * Gamma[m][i1] / (abs(np.dot(H[m][i1].conj(),w[m][i1]))**2)    
    p_next[i2] = I_OMA(M,K,N,H,G,w,S,p,m,i2) * Gamma[m][i2] / (abs(np.dot(H[m][i2].conj(),w[m][i2]))**2)
           
    return p_next

# OMA : there is 2 time slots, user 1 and 3 use time slot 1, user 2 and 4 use time slot 2.
# Same king of distributed iterative power control in OMA 
# At each iteration, each BS chooses its transmit power to meet its SINR constraints
# with the lowest feasible power and assuming powers from the other BS are fixed 
def algoOMA(M,K,N,H,G,S,Gamma,epsilon,iterations=100):
    # Get the beamforming vector
    w = w_mf(M,K,N,H)
    p_prev = np.ones((M,K))*np.inf
    p_next = np.zeros((M,K))
    count = 0
    while la.norm((p_next-p_prev).flatten(),1) > epsilon and count < iterations:
        count += 1
        p_prev = np.array(p_next)
        for m in range(M):
            # update p_next
            p_next[m] = f_OMA(M,K,N,H,G,w,S,Gamma,p_prev,m)

#        print(p_next)
        
    return p_next, count

# Check if the power allocation p is feasible with respect to the SINR constraints Gamma
def isFeasibleOMA(M,K,N,H,G,S,Gamma,p,epsilon):
    # Get the beamforming vector
    w = w_mf(M,K,N,H)
    # Compute the result
    res = True
    for m in range(M):
        res = res and (p[m][0]*(abs(np.dot(H[m][0].conj(),w[m][0]))**2)/I_OMA(M,K,N,H,G,w,S,p,m,0) - Gamma[m][0] > -epsilon) \
                  and (p[m][1]*(abs(np.dot(H[m][1].conj(),w[m][1]))**2)/I_OMA(M,K,N,H,G,w,S,p,m,1) - Gamma[m][1] > -epsilon)
    return res

'''
# Set seed for tests
np.random.seed(0)

N = 3 # Number of transmitter antenna
K = 2 # Number of users
M = 2 # Number of cells
H = (np.random.randn(M,K,N) + 1j*np.random.randn(M,K,N)) / math.sqrt(2) # link gain matrix
G = (np.random.randn(M,K,N) + 1j*np.random.randn(M,K,N)) / math.sqrt(2) # inter-cell link gain matrix
print('H[0]\n',H[0],'\n')
print('H[1]\n',H[1],'\n')
print('G[0]\n',G[0],'\n')
print('G[1]\n',G[1],'\n')
    
P = np.ones((M,K))
sigma_square = 1
# TEST beamforming vectors
print('W_MF\n',w_mf(M,K,N,H))
print()
print('W_ZF\n',w_zf(M,K,N,H))
print()
print('W_MSSE\n',w_mmse(M,K,N,H,P,sigma_square))

print()
S = np.identity(K)*1
'''
# TEST function I()
#i = 0
#m = 0
##p = np.zeros((M,K))
#p = np.ones((M,K))
#print(I(M,K,N,H,G,w_mf,S,p,m,i))
'''
Gamma = np.ones((M,K))*0.71132
epsilon = 1e-5
p_mf, pi_mf, count = algoNOMA(M,K,N,H,G,w_mf,S,Gamma,epsilon)
# TEST final condition
print('final pi',pi_mf)
print('count',count)
print('\np final\n',p_mf)
print('sum p final',la.norm(p_mf.flatten(),1))

for m in range(M):
    print('\nCell',m,':')
    print('\ntheta1',theta1(M,K,N,H,G,w_mf,S,p_mf,pi_mf,m))
    print('\ngamma12',gamma12(M,K,N,H,G,w_mf,S,p_mf,pi_mf,m))
    print('\ngamma2',gamma2(M,K,N,H,G,w_mf,S,p_mf,pi_mf,m))
    
if count >= 100:
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
'''
# TEST f
#m = 0
#pi = [[0,1],[0,1]]
#p_next = np.array(p)
#p_next[m] = f(M,K,N,H,G,w_mf,S,Gamma,pi[m],p,m)
#print(p_next)
#print('\ntheta1',theta1(M,K,N,H,G,w_mf,S,p_next,pi,m))
#print('\ngamma12',gamma12(M,K,N,H,G,w_mf,S,p_next,pi,m))
#print('\ngamma2',gamma2(M,K,N,H,G,w_mf,S,p_next,pi,m))
"""