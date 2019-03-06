#!/usr/bin/env python
# coding: utf-8

# # Question 1
# 
# ### Importing required libraries

# In[1]:


import numpy as np
import skimage as sk
from PIL import Image
import matplotlib.pyplot as plt


# ### Opening and processing image.
#     - Used default cat image `chelsea` included in scikit-image library.
#     - Converted it into greyscale.
#     - Resized the image to 32x32.

# In[2]:


input_img = sk.data.chelsea()
# input_img = Image.open('/home/sagar/Downloads/seven.jpeg')
# input_img = input_img.resize((32, 32))
# input_img = np.array(input_img)
input_img = sk.color.rgb2grey(input_img)
input_img = sk.transform.resize(input_img, (32, 32), anti_aliasing = True, mode = 'reflect')
input_img = sk.img_as_ubyte(input_img)
# print(input_img.dtype)
# input_img_2 = np.reshape(input_img,(3,32,32))
# input_img = input_img_2
im = Image.fromarray(input_img)
im = im.resize((312, 312))
im.show()
# for i in range(3):
#     im = Image.fromarray(input_img[i])
#     im = im.resize((312, 312))
#     im.show()
input_img.shape = (1, 32, 32)
input_img.shape


# ### Class defining Convolution Layer.

# In[3]:


class conv_layer:
    def __init__(self, channels, filters, filter_size, stride):
        self.channels = channels
        self.filters = filters
        self.filter_size = filter_size
        self.stride = stride
        
        self.weights = np.zeros((self.filters, self.channels, self.filter_size, self.filter_size))
        self.bias = np.random.rand(self.filters, 1)
#         self.bias = np.zeros((self.filters, 1))
        
        for i in range(self.filters):
            self.weights[i,:,:,:] = np.random.normal(loc = 0, scale = np.sqrt(1./(self.channels*self.filter_size*self.filter_size)), size = (self.channels, self.filter_size, self.filter_size))
            
    def forward(self, inp):
        self.c, self.w, self.h = inp.shape
        new_w = (self.w - self.filter_size)//self.stride + 1
        new_h = (self.h - self.filter_size)//self.stride + 1
        
        feature_map = np.zeros((self.filters, new_w, new_h))
#         print(self.weights)
#         print()
        for f in range(self.filters):
            for i in range(new_w):
                for j in range(new_h):
                    feature_map[f, i, j] = np.sum(inp[:, i:i+self.filter_size, j:j+self.filter_size] * self.weights[f,:,:,:]) + self.bias[f]
        
        return feature_map


# ### Class definig Max_Pooling Layer

# In[4]:


class pool_layer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, inp):
        self.c, self.w, self.h = inp.shape
        new_w = (self.w - self.pool_size)//self.stride + 1
        new_h = (self.h - self.pool_size)//self.stride + 1
        
        feature_map = np.zeros((self.c, new_w, new_h))
        for c in range(self.c):
            for i in range(self.w//self.stride):
                for j in range(self.w//self.stride):
                    feature_map[c, i, j] = np.max(inp[c, i*self.stride:i*self.stride+self.pool_size, j*self.stride:j*self.stride+self.pool_size])
        return feature_map


# ### Classs defining Fully Connected Layer.

# In[5]:


class fully_connected_layer:
    def __init__(self, nodes_curr_layer, nodes_next_layer):
        self.nodes_curr_layer = nodes_curr_layer
        self.nodes_next_layer = nodes_next_layer
        
        self.weight = np.random.rand(self.nodes_curr_layer, self.nodes_next_layer)
        self.bias = np.random.rand(1, self.nodes_next_layer)
    
    def forward(self, inp):
        self.mul = inp @ self.weight
        self.Z = np.add(self.mul, self.bias)
        return self.Z


# In[6]:


class flatten:
    def __init__(self):
        pass
    def forward(self, inp):
        self.c, self.w, self.h = inp.shape
        return inp.reshape(1, self.c * self.w * self.h)


# In[7]:


class activation_func:
    def __init__(self, func):
        self.func = func
    
    def forward(self, inp):
        if self.func == 'relu':
            inp[inp < 0] = 0
            inp[inp > 255] = 255
            return inp
        elif self.func == 'sigmoid':
            return 1/1 + np.exp(inp - np.max(inp))
        elif self.func == 'softmax':
            exp = np.exp(inp, dtype = np.float)
            return exp/np.sum(exp)


# ### Driver class
#     - It defines the architecture of LeNet 5.

# In[8]:


class conv_neural_net:
    def __init__(self):
        self.layers = []
        self.layers.append(conv_layer(1, 6, 5, 1))
        self.layers.append(activation_func('relu'))
        self.layers.append(pool_layer(2, 2))
        self.layers.append(conv_layer(6, 16, 5, 1))
        self.layers.append(activation_func('relu'))
        self.layers.append(pool_layer(2, 2))
        self.layers.append(conv_layer(16, 120, 5, 1))
        self.layers.append(activation_func('relu'))
        self.layers.append(flatten())
        self.layers.append(fully_connected_layer(120, 84))
        self.layers.append(activation_func('relu'))
        self.layers.append(fully_connected_layer(84, 2))
        self.layers.append(activation_func('relu'))
        
        self.num_layers = len(self.layers)
        
    def fit(self, inp):
        for l in range(self.num_layers):
            print(l)
            out = self.layers[l].forward(inp)
            inp = out
            if l < 1:
                for i in range(out.shape[0]):
    #                 out_img = sk.img_as_ubyte(out)
                    im = Image.fromarray(out[i])
                    im = im.resize((312, 312))
#                     plt.imshow(out[i])
                    im.show()
        return out


# In[9]:


cnn = conv_neural_net()
cnn.fit(input_img)


# # Question 2

# ### Part 1
#     - 156 parameters in 1st convolution layer.
# 
# ### Part 2
#     - There are no parameters on pooling layer.
#     
# ### Part 3
#     - Fully Connected layer has most number of paramters. (approx. 4800)
# 
# ### Part 4
#     - Fully connected layers take most amount of memory.

# In[ ]:




