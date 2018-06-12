# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 00:06:46 2018
@author: Luan Dong

Reference:
    Approximate Parameter Learning in Conditional Random Fields: An Empirical Investigation
    Filip Korˇc and Wolfgang F¨orstner

"""

import os
import numpy as np
#import chainer
import chainer.functions as F
from chainer import Variable
from scipy.misc import imread
import matplotlib.pyplot as plt
import maxflow # PyMaxflow

np.random.seed(101)    


#%% define model
class CRF(object):
    def __init__(self, width=64):
        self.w = Variable(np.random.normal(0,0.1,(1,2)).astype(np.float32))
        self.wv = np.zeros((1,2), dtype=np.float32) # velocity for momentum update
        
        self.v = Variable(np.random.normal(0,0.1,(1,2)).astype(np.float32))
        self.vv = np.zeros((1,2), dtype=np.float32) # velocity for momentum update
        
        self.L = width
        self.labels = [-1, 1]
        self.mmt = 0.95 # momentum
        
        
    def unary_feature(self, x): # x's shape is (1,64,64)
        bias = Variable(np.ones_like(x, dtype=np.float32))
        return F.concat((bias, x), axis=0)
    
    
    def unary_potential(self, x, y):
        ufeature = self.unary_feature(x)
        weighted_feature = F.matmul(self.w, ufeature.reshape((2,-1)))
        return F.absolute(F.tanh(weighted_feature.reshape((1,64,64))) - y)
    
    
    def pairwise_feature_h(self, x):
        # horizontal
        hbias = np.ones((1, self.L, self.L-1), dtype=np.float32)
        hfeature = F.absolute(x[:,:,:-1] - x[:,:,1:])
        return F.concat((hbias, hfeature), axis=0)
    
    
    def pairwise_feature_v(self, x):
        # vertical
        vbias = np.ones((1, self.L-1, self.L), dtype=np.float32)
        vfeature = F.absolute(x[:,:-1,:] - x[:,1:,:])
        return F.concat((vbias, vfeature), axis=0)
    
    
    def pairwise_potential_h(self, x, y):
        pfeature_h = self.pairwise_feature_h(x)
        weighted_f = F.matmul(self.v, pfeature_h.reshape((2,-1)))
        Ih = F.absolute(y[:,:,:-1] - y[:,:,1:]) * F.relu(weighted_f.reshape((1,64,63)))
        Ih_padr = F.concat((Ih, np.zeros((1,64,1),dtype=np.float32)), axis=2)
        Ih_padl = F.concat((np.zeros((1,64,1),dtype=np.float32), Ih), axis=2)
        return Ih_padr + Ih_padl
    
    
    def pairwise_potential_v(self, x, y):
        pfeature_v = self.pairwise_feature_v(x)
        weighted_f = F.matmul(self.v, pfeature_v.reshape((2,-1)))
        Iv = F.absolute(y[:,:-1,:] - y[:,1:,:]) * F.relu(weighted_f.reshape((1,63,64)))
        Iv_padu = F.concat((np.zeros((1,1,64),dtype=np.float32), Iv), axis=1)
        Iv_padd = F.concat((Iv, np.zeros((1,1,64),dtype=np.float32)), axis=1)
        return Iv_padu + Iv_padd
    
    
    def graphcut_weights(self, x, y):
        pfeature_h = self.pairwise_feature_h(x)
        weighted_f = F.matmul(self.v, pfeature_h.reshape((2,-1)))
        Ih = F.absolute(y[:,:,:-1] - y[:,:,1:]) * F.relu(weighted_f.reshape((1,64,63)))
        Ih_padr = F.concat((Ih, np.zeros((1,64,1),dtype=np.float32)), axis=2)
        Ih_padl = F.concat((np.zeros((1,64,1),dtype=np.float32), Ih), axis=2)
        
        pfeature_v = self.pairwise_feature_v(x)
        weighted_f = F.matmul(self.v, pfeature_v.reshape((2,-1)))
        Iv = F.absolute(y[:,:-1,:] - y[:,1:,:]) * F.relu(weighted_f.reshape((1,63,64)))
        Iv_padu = F.concat((np.zeros((1,1,64),dtype=np.float32), Iv), axis=1)
        Iv_padd = F.concat((Iv, np.zeros((1,1,64),dtype=np.float32)), axis=1)
        
        return Ih_padr.data, Ih_padl.data, Iv_padu.data, Iv_padd.data
    
    
    def site_I(self, x, y, t):
        yOdd = y.data.copy()
        yOdd[:,0:64:2,0:64:2] = t
        maskOdd = np.zeros_like(yOdd, dtype=np.float32)
        maskOdd[:,0:64:2,0:64:2] = 1
        Iodd = self.pairwise_potential_h(x, yOdd) + self.pairwise_potential_v(x, yOdd)
        
        yEven = y.data.copy()
        yEven[:,1:64:2,1:64:2] = t
        maskEven = np.zeros_like(yEven, dtype=np.float32)
        maskEven[:,1:64:2,1:64:2] = 1
        Ieven = self.pairwise_potential_h(x, yEven) + self.pairwise_potential_v(x, yEven)
        
        return Iodd*Variable(maskOdd) + Ieven*Variable(maskEven)
    
    
    # approximate the partition function
    def log_Z(self, x, y):
        A0 = self.unary_potential(x, Variable( - np.ones_like(x.data, dtype=np.float32)))
        I0 = self.site_I(x,y,-1)
        
        A1 = self.unary_potential(x, Variable(np.ones_like(x.data, dtype=np.float32)))
        I1 = self.site_I(x,y,1)
        
        return F.logsumexp(F.concat((A0+I0, A1+I1), axis=0), axis=0)
        
            
    # pseudo log-likelihood
    def ll(self, x, y):
        # compute unary potential
        A = self.unary_potential(x, y)
        # computer pairwise potential
        I = self.pairwise_potential_h(x, y) + self.pairwise_potential_v(x, y)
        # compute partition function (approximately)
        Z = self.log_Z(x, y)
        return F.reshape(F.sum(A + I + F.expand_dims(Z, axis=0)), (1,1))
                
                

#%% read 64×64 binary images, create the dataset
img_list = os.listdir('imgs')
img_list = np.random.permutation(img_list)

# training set
trainx = np.empty((120,64,64), dtype=np.float32)
trainy = np.empty((120,64,64), dtype=np.float32)

for i in range(6):
    im = imread(os.path.join('imgs', img_list[i]))
    im = im[:,:,0]
    im = (im/255 - 0.5)*2 # pixels in {-1, 1}
    for j in range(20):
        # create 20 noisy images for each clean image        
        noise = np.random.normal(scale=0.5, size=im.shape)
        trainx[i*20+j,:,:] = im + noise
        trainy[i*20+j,:,:] = im

# testing set
testx = np.empty((20,64,64), dtype=np.float32)
testy = np.empty((20,64,64), dtype=np.float32)

for i in range(2):
    im = imread(os.path.join('imgs', img_list[i+5]))
    im = im[:,:,0]
    im = (im/255 - 0.5)*2 # pixels in {-1, 1}
    for j in range(10):
        noise = np.random.normal(scale=0.5, size=im.shape)
        testx[i*10+j,:,:] = im + noise
        testy[i*10+j,:,:] = im


#%% training
n_iter = 1
lr = 0.0001 # learning rate
        
# create model
model = CRF()   

training_loss = []
for i in range(n_iter):
    idx = np.random.permutation(120)
    for j in range(120):
        data = Variable(trainx[idx[j]].reshape((1,64,64)))
        label = Variable(trainy[idx[j]].reshape((1,64,64)))
        loss = model.ll(data, label)
        training_loss.append(loss.data.flatten())
        
        # clear gradient
        model.w.cleargrad()
        model.v.cleargrad()
        
        # compute gradient
        loss.backward()
        
        # gradient descent with momentum
        model.wv = model.wv * model.mmt - lr * model.w.grad
        model.w.data += model.wv
        
        model.vv = model.vv * model.mmt - lr * model.v.grad
        model.v.data += model.vv


# show the learning curve
plt.figure()
plt.plot(training_loss, 'b-')
plt.xlim(1, n_iter*120)
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('learning curve')
plt.show()

        
#%% testing
x = testx[0:1,:,:] # test the 1st noisy image
y = np.ones_like(x, dtype=np.float32)
y[:,0:64:2,0:64:2] = -1
y[:,1:64:2,1:64:2] = -1
y = Variable(y)

# visit https://pmneila.github.io/PyMaxflow/tutorial.html#binary-image-restoration
# for PyMaxflow usage

# build grid-structure CRF
g = maxflow.Graph[float]()
nodeids = g.add_grid_nodes(x[0].shape)

# Add non-terminal edges
Ih_padr, Ih_padl, Iv_padu, Iv_padd = model.graphcut_weights(x,y)
structure_l = np.array([[0, 0, 0],\
                        [0, 0, 1],\
                        [0, 0, 0]], dtype=np.float32)
g.add_grid_edges(nodeids, weights=Ih_padr[0], structure=structure_l, symmetric=False)

structure_r = np.array([[0, 0, 0],\
                        [1, 0, 0],\
                        [0, 0, 0]], dtype=np.float32)
g.add_grid_edges(nodeids, weights=Ih_padl[0], structure=structure_r, symmetric=False)

structure_u = np.array([[0, 1, 0],\
                        [0, 0, 0],\
                        [0, 0, 0]], dtype=np.float32)
g.add_grid_edges(nodeids, weights=Iv_padu[0], structure=structure_u, symmetric=False)

structure_d = np.array([[0, 0, 0],\
                        [0, 0, 0],\
                        [0, 1, 0]], dtype=np.float32)
g.add_grid_edges(nodeids, weights=Iv_padd[0], structure=structure_d, symmetric=False)

# It is found the above non-terminal edges are all zeros.
# So we can assignment a constant value instead.
#g.add_grid_edges(nodeids, 1)

# Edges for the source and sink nodes
A0 = model.unary_potential(x,Variable( - np.ones_like(x, dtype=np.float32)))
A1 = model.unary_potential(x,Variable(np.ones_like(x, dtype=np.float32)))
g.add_grid_tedges(nodeids, A1.data[0], A0.data[0])

# segmenting
g.maxflow()
sgm = g.get_grid_segments(nodeids)

rec_img = np.int_(sgm)
plt.imshow(rec_img, cmap='gray')
plt.title('denoised image')
plt.show()

plt.imshow(x[0], cmap='gray')
plt.title('original image')
plt.show()

