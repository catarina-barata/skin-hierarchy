# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 16:46:05 2020

@author: acfba
"""



import tensorflow as tf
from tensorflow.contrib import slim


def attention(data,state,dim_hidden,ratio,is_training):
        
    state_expand = tf.expand_dims(state,1)
    
    if ratio:
        #Channel attention
        states_score = slim.fully_connected(state_expand,dim_hidden//ratio,activation_fn=None,scope='score_state_ch')
        
        data_ch = tf.reshape(data,[int(data.shape[0]),9,9,int(data.shape[2])])
    
        scale = channel_attention(data_ch,name='cha_att', ratio=ratio)
        
        scale = tf.nn.tanh(tf.reshape(scale,[int(scale.shape[0]),-1,int(scale.shape[3])])+states_score)
        
        scale = tf.nn.sigmoid(slim.fully_connected(scale,dim_hidden,activation_fn=None,normalizer_fn=None,scope='ch_weights'))
        
        data = data*scale
        
    #Spatial attention
    states_score = slim.fully_connected(state_expand,dim_hidden,activation_fn=None,scope='score_state_sp')
    
    #Classical
    score = tf.nn.tanh(data + states_score)
    
    score = slim.dropout(score,0.8,is_training=is_training)
    
    #Compute Attention weights
    #1 - Classical
    att_weights = tf.nn.softmax(slim.fully_connected(score,1,activation_fn=None,normalizer_fn=None,scope='att_weights'),axis=1)
    
    #Get context
    context = att_weights*data
    context = tf.reduce_sum(context,axis=1)
    
    return context,att_weights#,context2,att_weights2

def channel_attention(input_feature, name='cha_att', ratio=8):
  
  kernel_initializer = tf.contrib.layers.variance_scaling_initializer()
  bias_initializer = tf.constant_initializer(value=0.0)
  
  with tf.variable_scope(name):
    
    channel = input_feature.get_shape()[-1]
        
    avg_pool = tf.reduce_mean(input_feature, axis=[1,2], keepdims=True)
        
    assert avg_pool.get_shape()[1:] == (1,1,channel)
    avg_pool = tf.layers.dense(inputs=avg_pool,
                                 units=channel//ratio,
                                 activation=tf.nn.relu,
                                 kernel_initializer=kernel_initializer,
                                 bias_initializer=bias_initializer,
                                 name='mlp_0',
                                 reuse=None)   
    assert avg_pool.get_shape()[1:] == (1,1,channel//ratio)

    max_pool = tf.reduce_max(input_feature, axis=[1,2], keepdims=True)    
    assert max_pool.get_shape()[1:] == (1,1,channel)
    max_pool = tf.layers.dense(inputs=max_pool,
                                 units=channel//ratio,
                                 activation=tf.nn.relu,
                                 name='mlp_0',
                                 reuse=True)   
    assert max_pool.get_shape()[1:] == (1,1,channel//ratio)

    scale = avg_pool + max_pool
    
  return scale 
