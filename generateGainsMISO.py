# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 10:44:50 2017

@author: lsalaun
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt

test = False

# Generate K random points in a hexagon radius R (i.e. "big radius", circumradius) 
# centered at (cx,cy)
# output : array of dim [K][2], output[k] == kth user's coordinates
def randHexagon(K,R=1,rmin=0,center=[0,0]):
    cx = center[0]
    cy = center[1]
    vectors = np.array([(-1.,0),(.5,math.sqrt(3.)/2.),(.5,-math.sqrt(3.)/2.)])
    
    pts = np.zeros((K,2))
    
    for k in range(K):
        pt_x = cx
        pt_y = cy
        while distance([pt_x,pt_y],[cx,cy]) < rmin:
            idx = random.choice([0,1,2])
            v1 = vectors[idx]
            v2 = vectors[(idx+1)%3]
            x = random.random()
            y = random.random()
            pt_x = R*(x*v1[0]+y*v2[0]) + cx
            pt_y = R*(x*v1[1]+y*v2[1]) + cy
        pts[k][0] = pt_x
        pts[k][1] = pt_y
    return pts

# Distance between 2D coordinates X1 and X2
def distance(X1,X2):
    return math.sqrt((X2[0]-X1[0])**2 + (X2[1]-X1[1])**2)

# Generate 2 users uniformly in a hexagonal cell of circumradius R
# Do this for 2 cells and compute the intra-cell link gain matrix H, inter-cell G
# N : number of antennas
def generateGainsMISO(R,rmin,M,K,N):
    
    # R : circumradius
    # r = math.sqrt(3)/2*R : inradius
    r = math.sqrt(3)/2*R
    # Generate the position of 2 users in 1 hexagonal cell of circumradius R
    # and centered at (0,-r) for cell 1 and (0,r) for cell 2
    assert M==7,'The number of cells is not valid'
    centers = [[0,0],[3/2*R,r],[3/2*R,-r],[0,-2*r],[-3/2*R,-r],[-3/2*R,r],[0,2*r]]
    centersV1 = [[0,-4*r],[-3*R,-2*r],[-3*R,2*r],[0,4*r],[3*R,2*r],[3*R,-2*r]]
    centersV2 = [[-3*R,0],[-3/2*R,3*r],[3/2*R,3*r],[3*R,0],[3/2*R,-3*r],[-3/2*R,-3*r]]
    pos = np.zeros((M,K,2))
    for i in range(M):
#        print(randHexagon(K,R,rmin,centers[i]))
        pos[[i]] = [randHexagon(K,R,rmin,centers[i])]
    # print('pos: ',pos)

    # TEST 3 sector <-> 1 cell
#    BS1 = [R,-r]
#    BS2 = [-R,r]
#    centers = [BS1,BS2]

    H = np.zeros((M,M,K,N), dtype=complex)
    
    # i : index of the current cell
    # j : index of the interference cell
    
    for i in range(M):
        centersVirtual = centers[:]
        if i==1:
            centersVirtual[3]=centersV2[2]
            centersVirtual[4]=centersV2[3]
            centersVirtual[5]=centersV1[4]
        elif i==2:
            centersVirtual[4]=centersV2[3]
            centersVirtual[5]=centersV2[4]
            centersVirtual[6]=centersV1[5]
        elif i==3:
            centersVirtual[5]=centersV2[4]
            centersVirtual[6]=centersV2[5]
            centersVirtual[1]=centersV1[0]
        elif i==4:
            centersVirtual[6]=centersV2[5]
            centersVirtual[1]=centersV2[0]
            centersVirtual[2]=centersV1[1]
        elif i==5:
            centersVirtual[1]=centersV2[0]
            centersVirtual[2]=centersV2[1]
            centersVirtual[3]=centersV1[2]
        elif i==6:
            centersVirtual[2]=centersV2[1]
            centersVirtual[3]=centersV2[2]
            centersVirtual[4]=centersV1[3]

        for j in range(M):
        
            # We consider the K users in cell i and compute their direct gain and inter-cell gain
            # for the kth user in cell i, we compute the following
            # direct gain : H[i][k]
            # inter-cell gain : H[j][k], j!=i
    
            # Distance to cell i
            d_i = np.array([ distance(pos[i][k],centersVirtual[j]) for k in range(K)])
            # print('d: ',d_i)

            # TODO corriger /2 pour tous les termes??
            # Every term is /2 in dB (sqrt() in scalar W) because we are interested in the complex 
            # link gain matrix H such that y = h*x+noise
            
            # fast fading (smale-scale fading) -> independent rayleigh fading with variance 1
            # For each user, each antennas
            # rayleigh = (np.random.randn(2,N) + 1j*np.random.randn(2,N))/math.sqrt(2)
            # rayleigh = np.sqrt(np.random.randn(K,N) + 1j*np.random.randn(K,N))
            rayleigh = (np.random.randn(K,N) + 1j*np.random.randn(K,N))/math.sqrt(2)
            
            # Shadow fading for user k in cell i : (X_k + X_i)/sqrt(2)
            # where X_i is the common part in the cell and X_k is the individual part
            CommonShadowing = 10*np.random.randn(1)                             # SD 10 dB
            # shadowing = -((10*np.random.randn(K) + CommonShadowing)/math.sqrt(2))/2 # lognormal distributed with SD 10 dB  #TODO TODO
            shadowing = -(10*np.random.randn(K)+ CommonShadowing) # /math.sqrt(2)/2
            shadowing = np.power(10,(shadowing/10))                               # dB to scalar
            
            # path_loss_i : intra-cell path loss
            path_loss_i = -(128.1+37.6*np.log10(d_i/1000))/2   # path loss model : BUdistance/1000 in km
            path_loss_i = np.power(10,(path_loss_i/10))        # dB to scalar
            
            # path_loss_j : inter-cell path loss
            # path_loss_j = -(128.1+37.6*np.log10(d_j/1000))/2   # path loss model : BUdistance/1000 in km
            # path_loss_j = np.power(10,(path_loss_j/10))        # dB to scalar                    
            # print('path loss: ',path_loss_i,' rayleigh: ',rayleigh,'shadowing: ',shadowing)
            # H[i,j,:,:] = [[ path_loss_i[k] * rayleigh[k][n] * shadowing[k] for n in range(N)] for k in range(K)]
            H[i,j,:,:] = [[ path_loss_i[k] * rayleigh[k][n] * shadowing[k] for n in range(N)] for k in range(K)]
            # G[j] = [[ path_loss_j[k] * rayleigh[k][n] * shadowing[k] for n in range(N)] for k in range(K)]
        
        
#        print('cell',i)
#        print('positions\n',pos[i])
#        print('d_i',d_i)   
#        print('d_j',d_j)
#        print('rayleigh\n',rayleigh) 
#        print('path_loss_i',path_loss_i) 
#        print('path_loss_j',path_loss_j) 
#        print('shadowing\n',shadowing) 
#        print('H[i]\n',H[i])
#        print('G[j]\n',G[j])
            
    global test
    if test:
        colors = ['red','brown','coral','gold','green','pink','blue']
        plt.figure()
        for i in range(M):
            plt.scatter([pos[i][k][0] for k in range(K)],[pos[i][k][1] for k in range(K)],color=colors[i],marker='.')
            plt.plot(centers[i][0],centers[i][1],color=colors[i],marker='o')
        plt.xlim(-5/2*R,5/2*R)
        plt.ylim(-5/2*R,5/2*R)
        plt.grid()
        plt.show()

    
    return H

# Test randHexagon
#K = 5000
#R = 250
#rmin = 35
#r = R*math.sqrt(3)/2
## TEST 1
##tmp = randHexagon(K,R,rmin,0,0)
##plt.plot([x[0] for x in tmp],[x[1] for x in tmp],'.')
##plt.plot([0,0,-rmin,rmin],[-rmin,rmin,0,0],'ro')
## TEST 2
#tmp = np.append(randHexagon(K,R,rmin,0,-r),randHexagon(K,R,rmin,0,r),axis = 0)
#plt.plot([x[0] for x in tmp],[x[1] for x in tmp],'.')
#plt.plot([R,-R],[-r,r],'ro')
