# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 12:13:54 2017

@author: lsalaun
"""

import math
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import generateGainsMISO
import MISO_NOMA
from scipy import stats
import pandas as pd
from pandas import DataFrame,Series


R = 1000 # hexagonal cell circumradius in m
rmin = 35
N = 2 # number of antennas
M = 7 # number of cells
K = 2 # number of users per cell
W = 10e6 # total bandwidth in Hz
alpha = 0.5 # semiorthogonal factor
Noise0 = -174 # noise density dBm/Hz
Noise = (10**((Noise0-30)/10))*W # noise power in W (10**((Noise0-30)/10)) : conversion dBm -> W)

np.random.seed(1)
H = generateGainsMISO.generateGainsMISO(R,rmin,M,K,N)
print('\nH: ',H.shape)

S = np.ones((M,K))*Noise
DataRateRequirement = 15e4 #15e6 #bit/s
Gamma = np.ones((M,K))*((2**(DataRateRequirement/W))-1)

epsilon = 1e-6 # 1e-6
iterations = 100


print('\n------------------------ algoNOMA_disjoint ------------------------')
print('Here the user clustering and power control are done seperately')
print('The system can converge according to Yates framework')

q_disjoint, p_disjoint, pi_disjoint, count, isFeasible = MISO_NOMA.algoNOMA_disjoint(M,K,N,H,MISO_NOMA.w_zf,S,Gamma,alpha,epsilon,iterations)
print('number of iterations: ', count)
print('sum Power: ', sum(sum(q_disjoint)))
print('pi_disjoint: ', pi_disjoint)
print('p final feasible?',isFeasible)

print('\n------------------------ algoNOMA_InterferenceAware ------------------------')
print('Here the user clustering and power control are done jointly')
print('The system can converge according to Yates framework')
q_IAware, p_IAware, pi_IAware, count, isFeasible = MISO_NOMA.algoNOMA_IAware(M,K,N,H,MISO_NOMA.w_zf,S,Gamma,alpha,epsilon,iterations)
print('number of iterations: ', count)
# print('q_joint: ', q_joint)
print('sum Power: ', sum(sum(q_IAware)))
print('pi_IAware: ', pi_IAware)
print('p final feasible?',isFeasible)

print('\n------------------------ algoNOMA_PowerAware ------------------------')
print('Here the user clustering and power control are done jointly')
print('The system can converge according to Yates framework')
q_PAware, p_PAware, pi_PAware, count, isFeasible = MISO_NOMA.algoNOMA_PAware(M,K,N,H,MISO_NOMA.w_zf,S,Gamma,alpha,epsilon,iterations)
print('number of iterations: ', count)
# print('q_joint: ', q_joint)
print('sum Power: ', sum(sum(q_PAware)))
print('pi_PAware: ', pi_PAware)
print('p final feasible?',isFeasible)


print('\n------------------------ algoOMA ------------------------')
Gamma_oma = np.ones((M,K))*((2**(2*DataRateRequirement/W))-1)
p_oma_zf, count, isFeasible = MISO_NOMA.algoOMA(M,K,N,H,MISO_NOMA.w_zf_OMA,S,Gamma,epsilon,iterations)
print('number of iterations: ', count)
# print('q_joint: ', q_joint)
print('sum Power: ', sum(sum(p_oma_zf)))
print('p final feasible?',isFeasible)

print('\n------------------------ algoNOMA_best ------------------------')
q_best, p_best, pi_best, count, isFeasible = MISO_NOMA.algoNOMA_best(M,K,N,H,MISO_NOMA.w_zf,S,Gamma,alpha,epsilon,iterations)
print('number of iterations: ', count)
# print('q_joint: ', q_joint)
print('sum Power: ', sum(sum(q_best)))
print('pi_best: ', pi_best)
print('p final feasible?',isFeasible)
"""

print('\n------------------------ Simulaions to compare different K ------------------------')
print('The sum power consumption relative to the number of users per cell.')
Ks = range(2,5,1)
nDrops = 1000

P_disjoint = np.zeros(len(Ks))
P_IAware = np.zeros(len(Ks))
P_PAware = np.zeros(len(Ks))
P_OMA = np.zeros(len(Ks))
P_best = np.zeros(len(Ks))

nFeasible_min = np.zeros(len(Ks))
nOutage_disjoint = np.zeros(len(Ks))
nOutage_IAware = np.zeros(len(Ks))
nOutage_PAware = np.zeros(len(Ks))
nOutage_OMA = np.zeros(len(Ks))
nOutage_best = np.zeros(len(Ks))

nIter_disjoint = np.zeros(len(Ks))
nIter_IAware = np.zeros(len(Ks))
nIter_PAware = np.zeros(len(Ks))
nIter_OMA = np.zeros(len(Ks))
nIter_best = np.zeros(len(Ks))

P_disjoint_k = np.zeros((len(Ks),nDrops,M))
P_IAware_k = np.zeros((len(Ks),nDrops,M))
P_PAware_k = np.zeros((len(Ks),nDrops,M))
P_OMA_k = np.zeros((len(Ks),nDrops,M))
P_best_k = np.zeros((len(Ks),nDrops,M))
index_Feasible = np.zeros((len(Ks),nDrops))

for idxK in range(len(Ks)):
    K = Ks[idxK]
    print('Number of users per cell: ',K)
    nIter_disjoint_k = np.zeros(nDrops)
    nIter_IAware_k = np.zeros(nDrops)
    nIter_PAware_k = np.zeros(nDrops)
    nIter_OMA_k = np.zeros(nDrops)
    nIter_best_k = np.zeros(nDrops)
    for Drop in range(nDrops):
        # print('Drop: ',Drop)
        H = generateGainsMISO.generateGainsMISO(R,rmin,M,K,N)
        S = np.ones((M,K))*Noise
        DataRateRequirement = 15e4 #15e6 #bit/s
        Gamma = np.ones((M,K))*((2**(DataRateRequirement/W))-1)
        Gamma_oma = np.ones((M,K))*((2**(K*DataRateRequirement/W))-1)
        epsilon = 1e-6 # 1e-6
        iterations =100
        
        # Stand-alone user clustering
        q_disjoint, p_disjoint, pi_disjoint, count_disjoint, isFeasible_disjoint = MISO_NOMA.algoNOMA_disjoint(M,K,N,H,MISO_NOMA.w_zf,S,Gamma,alpha,epsilon,iterations)
        if isFeasible_disjoint == False:
            nOutage_disjoint[idxK] += 1
        else:
            nIter_disjoint_k[Drop] = count_disjoint
            P_disjoint_k[idxK,Drop,:] = np.sum(q_disjoint,axis=1)

        # Interference-aware user clustering
        q_IAware, p_IAware, pi_IAware, count_IAware, isFeasible_IAware = MISO_NOMA.algoNOMA_IAware(M,K,N,H,MISO_NOMA.w_zf,S,Gamma,alpha,epsilon,iterations)
        if isFeasible_IAware == False:
            nOutage_IAware[idxK] += 1
        else:
            nIter_IAware_k[Drop] = count_IAware
            P_IAware_k[idxK,Drop,:] = np.sum(q_IAware,axis=1)
            
        # Power-aware user clustering
        q_PAware, p_PAware, pi_PAware, count_PAware, isFeasible_PAware = MISO_NOMA.algoNOMA_PAware(M,K,N,H,MISO_NOMA.w_zf,S,Gamma,alpha,epsilon,iterations)
        if isFeasible_PAware == False:
            nOutage_PAware[idxK] += 1
        else:
            nIter_PAware_k[Drop] = count_PAware
            P_PAware_k[idxK,Drop,:] = np.sum(q_PAware,axis=1)
            
        # OMA
        p_OMA, count_OMA, isFeasible_OMA = MISO_NOMA.algoOMA(M,K,N,H,MISO_NOMA.w_zf_OMA,S,Gamma_oma,epsilon,iterations)
        if isFeasible_OMA == False:
            nOutage_OMA[idxK] += 1
        else:
            nIter_OMA_k[Drop] = count_OMA
            P_OMA_k[idxK,Drop,:] = np.sum(p_OMA,axis=1)
            
        # Best user clustering
        q_best, p_best, pi_best, count_best, isFeasible_best = MISO_NOMA.algoNOMA_best(M,K,N,H,MISO_NOMA.w_zf,S,Gamma,alpha,epsilon,iterations)
        if isFeasible_best == False:
            nOutage_best[idxK] += 1
        else:
            nIter_best_k[Drop] = count_best
            P_best_k[idxK,Drop,:] = np.sum(q_best,axis=1)
            
        if isFeasible_disjoint and isFeasible_IAware and isFeasible_PAware and isFeasible_OMA and isFeasible_best:
            index_Feasible[idxK,Drop] = 1
            nFeasible_min[idxK] += 1
            
    P_disjoint[idxK] = sum(np.multiply(np.sum(P_disjoint_k[idxK,:,:],axis=1),index_Feasible[idxK,:]))/nFeasible_min[idxK]
    P_IAware[idxK] = sum(np.multiply(np.sum(P_IAware_k[idxK,:,:],axis=1),index_Feasible[idxK,:]))/nFeasible_min[idxK]
    P_PAware[idxK] = sum(np.multiply(np.sum(P_PAware_k[idxK,:,:],axis=1),index_Feasible[idxK,:]))/nFeasible_min[idxK]
    P_OMA[idxK] = sum(np.multiply(np.sum(P_OMA_k[idxK,:,:],axis=1),index_Feasible[idxK,:]))/nFeasible_min[idxK]
    P_best[idxK] = sum(np.multiply(np.sum(P_best_k[idxK,:,:],axis=1),index_Feasible[idxK,:]))/nFeasible_min[idxK]
    nIter_disjoint[idxK] = sum(nIter_disjoint_k)/(nDrops-nOutage_disjoint[idxK])
    nIter_IAware[idxK] = sum(nIter_IAware_k)/(nDrops-nOutage_IAware[idxK])
    nIter_PAware[idxK] = sum(nIter_PAware_k)/(nDrops-nOutage_PAware[idxK])
    nIter_OMA[idxK] = sum(nIter_OMA_k)/(nDrops-nOutage_OMA[idxK])
    nIter_best[idxK] = sum(nIter_best_k)/(nDrops-nOutage_best[idxK])
    
plt.figure()
plt.plot(Ks,P_disjoint,label = 'Stand-alone user clustering')
plt.plot(Ks,P_IAware,label = 'Interference-aware user clustering')
plt.plot(Ks,P_PAware,label = 'Power-aware user clustering')
plt.plot(Ks,P_OMA,label = 'OMA')
plt.plot(Ks,P_best,label = 'SciPy user clustering')
plt.legend()
plt.title('Sum power consumption of 7 cells with different user clustering schemes, number of antennas N = '+str(N)+'')
plt.xlabel('Number of users per cell')
plt.ylabel('Transmit power (W)')
plt.grid()
plt.show()

all_data = [np.sum(P_IAware_k[i,:,:],axis=1) for i in range(0,3)]
fig = plt.figure(figsize=(8,6))
plt.boxplot(all_data,
            notch=False, # box instead of notch shape
            sym='rs',    # red squares for outliers
            vert=True)   # vertical box aligmnent
 
plt.xticks([y+1 for y in range(len(all_data))], ['2', '3', '4','5','6'])
plt.xlabel('Number of users per cell')
plt.ylabel('Sum power of 7 cells')
plt.grid()
t = plt.title('Box plot')
plt.show()


plt.figure()
plt.plot(Ks,nOutage_disjoint/nDrops,label = 'Stand-alone user clustering')
plt.plot(Ks,nOutage_IAware/nDrops,label = 'Interference-aware user clustering')
plt.plot(Ks,nOutage_PAware/nDrops,label = 'Power-aware user clustering')
plt.plot(Ks,nOutage_OMA/nDrops,label = 'OMA')
plt.plot(Ks,nOutage_best/nDrops,label = 'SciPy user clustering')
plt.legend()
plt.title('Outage rate of different user clustering schemes, number of antennas N = '+str(N)+'')
plt.xlabel('Number of users per cell')
plt.ylabel('Outage rate')
plt.grid()
plt.show()

plt.figure()
plt.plot(Ks,nIter_disjoint,label = 'Stand-alone user clustering')
plt.plot(Ks,nIter_IAware,label = 'Interference-aware user clustering')
plt.plot(Ks,nIter_PAware,label = 'Power-aware user clustering')
plt.plot(Ks,nIter_OMA,label = 'OMA')
plt.plot(Ks,nIter_best,label = 'SciPy user clustering')
plt.legend()
plt.title('Mean number of iterations of different user clustering schemes, number of antennas N = '+str(N)+'')
plt.xlabel('Number of users per cell')
plt.ylabel('Number of iterations')
plt.grid()
plt.show()      

def plotCDF(samples,numbins):
    res = stats.relfreq(samples, numbins) 
    x = res.lowerlimit + np.linspace(0, res.binsize*res.frequency.size,res.frequency.size) 
    y=np.cumsum(res.frequency)
    return x,y


k_test = 4
idxK = k_test-2
P_disjoint_x,P_disjoint_y = plotCDF(np.multiply(np.sum(P_disjoint_k[idxK,:,:],axis=1),index_Feasible[idxK,:]),10000)
P_IAware_x,P_IAware_y = plotCDF(np.multiply(np.sum(P_IAware_k[idxK,:,:],axis=1),index_Feasible[idxK,:]),10000)
P_PAware_x,P_PAware_y = plotCDF(np.multiply(np.sum(P_PAware_k[idxK,:,:],axis=1),index_Feasible[idxK,:]),10000)
P_OMA_x,P_OMA_y = plotCDF(np.multiply(np.sum(P_OMA_k[idxK,:,:],axis=1),index_Feasible[idxK,:]),10000)
P_best_x,P_best_y = plotCDF(np.multiply(np.sum(P_best_k[idxK,:,:],axis=1),index_Feasible[idxK,:]),10000)

plt.figure()
plt.plot(P_disjoint_x,P_disjoint_y,label = 'Stand-alone user clustering')
plt.plot(P_IAware_x,P_IAware_y,label = 'Interference-aware user clustering')
plt.plot(P_PAware_x,P_PAware_y,label = 'Power-aware user clustering')
plt.plot(P_OMA_x,P_OMA_y,label = 'OMA')
plt.plot(P_best_x,P_best_y,label = 'SciPy user clustering')
plt.legend()
plt.xlim(-1,5)
plt.ylim(0.7,1.001)
# plt.axis([0, 10, ymin, ymax])
# plt.yscale('logit')
# fig.set_xlim(0, 10)
# plt.set_ylim(1e-1, 1e3)
plt.title('CDF of power consumption of each cell with different user clustering schemes, number of antennas N = '+str(N)+' K = '+str(k_test)+'')
plt.xlabel('Power consumption/cell')
plt.ylabel('CDF')
plt.grid()
plt.show()

#plt.figure()
#plt.plot(stats.cumfreq(P_disjoint_k[0,:,:].flatten())[0],label = 'Stand-alone user clustering')
#plt.plot(stats.cumfreq(P_IAware_k[0,:,:].flatten())[0],label = 'Interference-aware user clustering')
#plt.plot(stats.cumfreq(P_PAware_k[0,:,:].flatten())[0],label = 'Power-aware user clustering')
#plt.plot(stats.cumfreq(P_OMA_k[0,:,:].flatten())[0],label = 'OMA')
#plt.plot(stats.cumfreq(P_best_k[0,:,:].flatten())[0],label = 'SciPy user clustering')
#plt.legend()
#plt.title('CDF of power consumption of each cell with different user clustering schemes, number of antennas N = '+str(N)+'')
#plt.xlabel('Power consumption/cell')
#plt.ylabel('Frequency')
#plt.grid()
#plt.show()


print('\n------------------------ Simulaions to compare ------------------------')
print('The sum power consumption relative to the number of users per cell.')
DataRateRequirements = list(range(1,10,2)) # [1e5,5e5,1e6,5e6,1e7] # range(10e4,10e6,5e4) #15e6 #bit/s
DataRateRequirements = np.multiply(DataRateRequirements,int(1e5))
nDrops = 100

P_disjoint = np.zeros(len(DataRateRequirements))
P_IAware = np.zeros(len(DataRateRequirements))
P_PAware = np.zeros(len(DataRateRequirements))
P_OMA = np.zeros(len(DataRateRequirements))

nOutage_disjoint = np.zeros(len(DataRateRequirements))
nOutage_IAware = np.zeros(len(DataRateRequirements))
nOutage_PAware = np.zeros(len(DataRateRequirements))
nOutage_OMA = np.zeros(len(DataRateRequirements))

nIter_disjoint = np.zeros(len(DataRateRequirements))
nIter_IAware = np.zeros(len(DataRateRequirements))
nIter_PAware = np.zeros(len(DataRateRequirements))
nIter_OMA = np.zeros(len(DataRateRequirements))

for idx in range(len(DataRateRequirements)):
    DataRateRequirement = DataRateRequirements[idx]
    print('Data Rate Requirement: ',DataRateRequirement)
    P_disjoint_R = np.zeros(nDrops)
    P_IAware_R = np.zeros(nDrops)
    P_PAware_R = np.zeros(nDrops)
    P_OMA_R = np.zeros(nDrops)
    nIter_disjoint_R = np.zeros(nDrops)
    nIter_IAware_R = np.zeros(nDrops)
    nIter_PAware_R = np.zeros(nDrops)
    nIter_OMA_R = np.zeros(nDrops)
    for Drop in range(nDrops):
        H = generateGainsMISO.generateGainsMISO(R,rmin,M,K,N)
        S = np.ones((M,K))*Noise
        # DataRateRequirement = 15e4 #15e6 #bit/s
        Gamma = np.ones((M,K))*((2**(DataRateRequirement/W))-1)
        Gamma_oma = np.ones((M,K))*((2**(K*DataRateRequirement/W))-1)
        epsilon = 1e-6 # 1e-6
        iterations = 100
        
        # Stand-alone user clustering
        q_disjoint, p_disjoint, pi_disjoint, count_disjoint, isFeasible_disjoint = MISO_NOMA.algoNOMA_disjoint(M,K,N,H,MISO_NOMA.w_zf,S,Gamma,alpha,epsilon,iterations)
        if isFeasible_disjoint == False:
            # print('Unfeasible stand-alone, Sum power: ',sum(sum(q_disjoint)))
            nOutage_disjoint[idx] += 1
            P_disjoint_R[Drop] = 0
        else:
            nIter_disjoint_R[Drop] = count_disjoint
            P_disjoint_R[Drop] = sum(sum(q_disjoint))

        # Interference-aware user clustering
        q_IAware, p_IAware, pi_IAware, count_IAware, isFeasible_IAware = MISO_NOMA.algoNOMA_IAware(M,K,N,H,MISO_NOMA.w_zf,S,Gamma,alpha,epsilon,iterations)
        if isFeasible_IAware == False:
            # print('Unfeasible IAware, Sum power: ',sum(sum(q_IAware)))
            nOutage_IAware[idx] += 1
            P_IAware_R[Drop] = 0
        else:
            nIter_IAware_R[Drop] = count_IAware
            P_IAware_R[Drop] = sum(sum(q_IAware))
            
        # Power-aware user clustering
        q_PAware, p_PAware, pi_PAware, count_PAware, isFeasible_PAware = MISO_NOMA.algoNOMA_PAware(M,K,N,H,MISO_NOMA.w_zf,S,Gamma,alpha,epsilon,iterations)
        if isFeasible_PAware == False:
            # print('Unfeasible PAware, Sum power: ',sum(sum(q_PAware)), 'Sum power IAware: ',sum(sum(q_IAware)))
            # print('\n pi_PAware: \n',pi_PAware)
            # print('\n pi_IAware: \n',pi_IAware)
            nOutage_PAware[idx] += 1
            P_PAware_R[Drop] = 0
        else:
            nIter_PAware_R[Drop] = count_IAware
            P_PAware_R[Drop] = sum(sum(q_PAware))
            
        # OMA
        p_OMA, count_OMA, isFeasible_OMA = MISO_NOMA.algoOMA(M,K,N,H,MISO_NOMA.w_zf_OMA,S,Gamma_oma,epsilon,iterations)
        if isFeasible_OMA == False:
            # print('Unfeasible IAware, Sum power: ',sum(sum(q_IAware)))
            nOutage_OMA[idx] += 1
            P_OMA_R[Drop] = 0
        else:
            nIter_OMA_R[Drop] = count_OMA
            P_OMA_R[Drop] = sum(sum(p_OMA))
            
    P_disjoint[idx] = sum(P_disjoint_R)/(nDrops-nOutage_disjoint[idx])
    P_IAware[idx] = sum(P_IAware_R)/(nDrops-nOutage_IAware[idx])
    P_PAware[idx] = sum(P_PAware_R)/(nDrops-nOutage_PAware[idx])
    P_OMA[idx] = sum(P_OMA_R)/(nDrops-nOutage_OMA[idx])
    nIter_disjoint[idx] = sum(nIter_disjoint_R)/(nDrops-nOutage_disjoint[idx])
    nIter_IAware[idx] = sum(nIter_IAware_R)/(nDrops-nOutage_IAware[idx])
    nIter_PAware[idx] = sum(nIter_PAware_R)/(nDrops-nOutage_PAware[idx])
    nIter_OMA[idx] = sum(nIter_OMA_R)/(nDrops-nOutage_OMA[idx])
    
plt.figure()
plt.plot(DataRateRequirements,P_disjoint,label = 'Stand-alone user clustering')
plt.plot(DataRateRequirements,P_IAware,label = 'Interference-aware user clustering')
plt.plot(DataRateRequirements,P_PAware,label = 'Power-aware user clustering')
plt.plot(DataRateRequirements,P_OMA,label = 'OMA')
plt.legend()
plt.title('Sum power of 7 cells with different user clustering schemes, number of antennas N = '+str(N)+', number of users per cell = '+str(K)+'')
plt.xlabel('Data Rate Requirement (bits/s)')
plt.ylabel('Transmit power (W)')
plt.grid()
plt.show()

plt.figure()
plt.plot(DataRateRequirements,nOutage_disjoint/nDrops,label = 'Stand-alone user clustering')
plt.plot(DataRateRequirements,nOutage_IAware/nDrops,label = 'Interference-aware user clustering')
plt.plot(DataRateRequirements,nOutage_PAware/nDrops,label = 'Power-aware user clustering')
plt.plot(DataRateRequirements,nOutage_OMA/nDrops,label = 'OMA')
plt.legend()
plt.title('Outage rate of different user clustering schemes, number of antennas N = '+str(N)+', number of users per cell = '+str(K)+'')
plt.xlabel('Data Rate Requirement (bits/s)')
plt.ylabel('Outage rate')
plt.grid()
plt.show()

plt.figure()
plt.plot(DataRateRequirements,nIter_disjoint,label = 'Stand-alone user clustering')
plt.plot(DataRateRequirements,nIter_IAware,label = 'Interference-aware user clustering')
plt.plot(DataRateRequirements,nIter_PAware,label = 'Power-aware user clustering')
plt.plot(DataRateRequirements,nIter_OMA,label = 'OMA')
plt.legend()
plt.title('Mean number of iterations of different user clustering schemes, number of antennas N = '+str(N)+', number of users per cell = '+str(K)+'')
plt.xlabel('Data Rate Requirement (bits/s)')
plt.ylabel('Number of iterations')
plt.grid()
plt.show()


print('\n------------------------ algoNOMA_bestResponse ------------------------')
print('Here the best decoding order is adopted at each iteration')
print('This may not converge in some cases (even if the problem is feasible)')
               
p_list, pi_mf, count = MISO_NOMA.algoNOMA_bestResponse_fullOutput(M,K,N,H,G,MISO_NOMA.w_mf,S,Gamma,epsilon,iterations)
p_mf = p_list[-1]

print('\nFinal decoding order pi =\n',pi_mf)
print('Number of iterations =',count)
print('p final =\n',p_mf)
print('sum p final =',la.norm(p_mf.flatten(),1))
print('p final feasible?',MISO_NOMA.isFeasibleNOMA(M,K,N,H,G,MISO_NOMA.w_mf,S,Gamma,pi_mf,p_mf,epsilon))

#for m in range(M):
#    print('\nCell',m,':')
#    print('theta1',MISO_NOMA.theta1(M,K,N,H,G,MISO_NOMA.w_mf,S,p_mf,pi_mf,m))
#    print('gamma12',MISO_NOMA.gamma12(M,K,N,H,G,MISO_NOMA.w_mf,S,p_mf,pi_mf,m))
#    print('gamma2',MISO_NOMA.gamma2(M,K,N,H,G,MISO_NOMA.w_mf,S,p_mf,pi_mf,m))

#print(p_list)
X = np.zeros((4,len(p_list)))
for i in range(len(p_list)):
    X[0][i] = p_list[i][0][0]
    X[1][i] = p_list[i][0][1]
    X[2][i] = p_list[i][1][0]
    X[3][i] = p_list[i][1][1]
   
fig = plt.figure()   
plt.plot(X[0]+X[1],'o-',label='BS 1')
plt.plot(X[2]+X[3],'o-',label='BS 2')
plt.legend()
plt.title('NOMA scheme convergence , N = '+str(N)+', $\Gamma$ = 10 Mbit/s')
plt.xlabel('Number of iterations')
plt.ylabel('Transmit power (W)')
plt.show()

print('\n------------------------ algoNOMA_allPossibilities ------------------------')
print('Here the iterative power control is applied 4 times with all 4 possible decoding orders')
print('The best solution among those 4 is returned at the end')
             
p_mf, pi_mf, count_mod = MISO_NOMA.algoNOMA_allPossibilities(M,K,N,H,G,MISO_NOMA.w_mf,S,Gamma,epsilon,iterations)

if p_mf is not None:
    print('\nFinal decoding order pi =\n',pi_mf)
    print('Number of iterations =',count_mod)
    print('p final =\n',p_mf)
    print('sum p final =',la.norm(p_mf.flatten(),1))
    print('p final feasible?',MISO_NOMA.isFeasibleNOMA(M,K,N,H,G,MISO_NOMA.w_mf,S,Gamma,pi_mf,p_mf,epsilon))
else:
    print('No feasible solution')
    
#for m in range(M):
#    print('\nCell',m,':')
#    print('theta1',MISO_NOMA.theta1(M,K,N,H,G,MISO_NOMA.w_mf,S,p_mf,pi_mf,m))
#    print('gamma12',MISO_NOMA.gamma12(M,K,N,H,G,MISO_NOMA.w_mf,S,p_mf,pi_mf,m))
#    print('gamma2',MISO_NOMA.gamma2(M,K,N,H,G,MISO_NOMA.w_mf,S,p_mf,pi_mf,m))

#print(p_list)

print('\n------------------------ algoNOMA_mix ------------------------')
print('Combines algoNOMA_bestResponse and fixed decoding order with some probability at each iteration')
print('This works well and seems to converge more often than algoNOMA (if the problem is feasible)')
print('However, the returned power is not always as good as algoNOMA_allPossibilities')
     
seed = 0
alpha = 0.99
       
p_mf, pi_mf, count_mix = MISO_NOMA.algoNOMA_mix(M,K,N,H,G,MISO_NOMA.w_mf,S,Gamma,epsilon,iterations,alpha=alpha,seed=seed)

if p_mf is not None:
    print('\nFinal decoding order pi =\n',pi_mf)
    print('Number of iterations =',count_mix)
    print('p final =\n',p_mf)
    print('sum p final =',la.norm(p_mf.flatten(),1))
    print('p final feasible?',MISO_NOMA.isFeasibleNOMA(M,K,N,H,G,MISO_NOMA.w_mf,S,Gamma,pi_mf,p_mf,epsilon))
else:
    print('No feasible solution')

print('\n------------------------ OMA ------------------------')

Gamma_oma = np.ones((M,K))*((2**(2*DataRateRequirement/W))-1)
print('\nGamma NOMA\n',Gamma)
print('Gamma OMA\n',Gamma_oma)
print()

p_oma_mf, count = MISO_NOMA.algoOMA(M,K,N,H,G,S,Gamma_oma,epsilon,iterations)

print('------- MF beamforming -------')
print('Number of iterations =',count)
print('p final =\n',p_oma_mf)
print('sum p final =',la.norm(p_oma_mf.flatten(),1))
print('p final feasible?',MISO_NOMA.isFeasibleOMA(M,K,N,H,G,S,Gamma_oma,p_oma_mf,epsilon*10))

print('\n------------------------ NOMA no SIC ------------------------')
print('Here we do not apply SIC, therefore there is no decoding order pi to consider')
print('We simulate different beamforming vectors: MF, ZF, MMSE')
print()

print('\n------- MF beamforming -------')
res = MISO_NOMA.algoNOMA_noSIC(M,K,N,H,G,MISO_NOMA.w_mf,S,Gamma,epsilon)
if res['x'] is not None:
    #print(res['x'])
    p_final = np.zeros((M,K))
    for m in range(M):
        for k in range(K):
            p_final[m][k] = res['x'][m*2+k]
    print('p_final =\n',p_final)
    print('sum p final =',np.sum(p_final))
    #print('success',res.success)
    print('p final feasible?',MISO_NOMA.isFeasibleNOMA_noSIC(M,K,N,H,G,MISO_NOMA.w_mf,S,Gamma,p_final,epsilon))
else:
    print('No feasible solution')
    
print('\n------- ZF beamforming -------')
res = MISO_NOMA.algoNOMA_noSIC(M,K,N,H,G,MISO_NOMA.w_zf,S,Gamma,epsilon)
if res['x'] is not None:
    #print(res['x'])
    p_final = np.zeros((M,K))
    for m in range(M):
        for k in range(K):
            p_final[m][k] = res['x'][m*2+k]
    print('p_final =\n',p_final)
    print('sum p final =',np.sum(p_final))
    #print('success',res.success)
    print('p final feasible?',MISO_NOMA.isFeasibleNOMA_noSIC(M,K,N,H,G,MISO_NOMA.w_zf,S,Gamma,p_final,epsilon))
else:
    print('No feasible solution')
    
print('\n------- MMSE beamforming -------')
res = MISO_NOMA.algoNOMA_noSIC(M,K,N,H,G,MISO_NOMA.w_mmse,S,Gamma,epsilon)
if res['x'] is not None:
    #print(res['x'])
    p_final = np.zeros((M,K))
    for m in range(M):
        for k in range(K):
            p_final[m][k] = res['x'][m*2+k]
    print('p_final =\n',p_final)
    print('sum p final =',np.sum(p_final))
    #print('success',res.success)
    print('p final feasible?',MISO_NOMA.isFeasibleNOMA_noSIC(M,K,N,H,G,MISO_NOMA.w_mmse,S,Gamma,p_final,epsilon))
else:
    print('No feasible solution')
    
"""