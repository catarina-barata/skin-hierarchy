# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:06:43 2019

@author: acfba
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import os.path
import sys
import numpy as np
import sklearn.metrics as metrics
from tqdm import tqdm


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf

from nets import densenet,vgg


from tensorflow.contrib import slim

from sklearn.feature_extraction.text import CountVectorizer

import data_import as da
import attention_modules as att


def get_initial_lstm(data,dim_hidden,is_training):
    initial_hidden = slim.fully_connected(slim.dropout(data,0.5,is_training = is_training),dim_hidden,activation_fn=tf.nn.tanh,scope = 'init_hidden_state')
    initial_memory = slim.fully_connected(slim.dropout(data,0.5,is_training = is_training),dim_hidden,activation_fn=tf.nn.tanh,scope = 'init_memory_state')
 
    return tf.concat([initial_hidden,initial_memory],1)

def apply_network_img(data_dense,batch_size,is_training_net,is_training_drop,net):
    
    if net=='D': #DenseNet-161
         with slim.arg_scope(densenet.densenet_arg_scope()):
             logits,_ = densenet.densenet161(data_dense,num_classes = None, 
                            is_training_batch=is_training_net,is_training_drop = is_training_drop)
                                       
    elif net=='V': #VGG-16    
         with slim.arg_scope(vgg.vgg_arg_scope()):
            logits,_ = vgg.vgg_16(data_dense,num_classes = None, is_training=is_training_drop,spatial_squeeze=False,
               classification = False, max_pool=False)
                            
    return logits
 
def apply_lstm(train_size,data,captions,weights,dim_hidden,dim_emb,dim_embed2,lstm,n_lstm_steps,n_words,wordtoix,embedding,class_inf,net):
    
    data = da.transform_data(data, is_training=True)
    
    data = apply_network_img(data, tf.shape(data)[0], is_training_net=False,
                             is_training_drop=False,net=net)
    
    mid = data.get_shape()[1]
    
    data = tf.reshape(data,[-1, mid*mid, int(data.shape[3])])
    
    state = get_initial_lstm(tf.reduce_mean(data,1),dim_hidden,True)

    current_caption_ind = tf.contrib.lookup.string_to_index(captions,wordtoix)
    
    embedding_map = tf.get_variable(
          name="map",
          shape=[n_words, dim_embed2],
          initializer= tf.constant_initializer(embedding),
          trainable=False
          )

    word_embedding = tf.zeros([tf.shape(data)[0], dim_embed2])
    
    total_loss = 0.0
    
    with tf.variable_scope("RNN",reuse=tf.AUTO_REUSE):
        
        image_embedding,att_weights = att.attention(data,state,int(data.shape[2]),Flags.ratio,1)
   
        current_embedding = tf.concat([image_embedding, word_embedding],axis=-1)  
        
        for i in range(1,n_lstm_steps-1):
             
                 _, state = lstm(current_embedding,state)
                                  
                 logits = slim.dropout(state,0.5,is_training=True,scope='drop')
                 
                 with tf.variable_scope("Caption",reuse=tf.AUTO_REUSE):
                            
                        #perform a softmax classification to generate the next word in the caption
                        
                        if class_inf == 1:
                            logit = slim.fully_connected(logits,n_words,activation_fn=None,normalizer_fn=None,scope='word_encoding')
                        else:
                            logit = slim.fully_connected(logits,dim_embed2,activation_fn=None,normalizer_fn=None,scope='state_encoding')
                                                                                
                            context = slim.fully_connected(image_embedding,dim_embed2,activation_fn=None,normalizer_fn=None,scope='context')
                            
                            logit += context
                            
                            logit = tf.nn.tanh(logit)
                            
                            logit = slim.dropout(logit, 0.5, is_training=True, scope='drop_combo')
                            
                            logit = slim.fully_connected(logit, n_words,activation_fn=None,normalizer_fn=None,scope='word_encoding')
##
                        onehot = tf.one_hot(tf.squeeze(tf.slice(current_caption_ind,[0,i],[tf.shape(data)[0],1]),1),n_words)
#                        
                        if i==1:
                            one_hot_path = onehot
                            probs_path = tf.nn.softmax(logit)
                        else:
                            one_hot_path += onehot
                        
                            probs_path += tf.nn.softmax(logit)
                        
                        weight = tf.matmul(onehot,tf.cast(tf.expand_dims(weights,1),tf.float32),transpose_b=False)
                        
                        weight = train_size/weight
                        
                        weight = tf.squeeze(weight/tf.reduce_mean(weight))
#                        
                        xentropy = tf.losses.softmax_cross_entropy(onehot,logits=logit, weights=1.0, loss_collection=None,reduction = tf.losses.Reduction.NONE)
                        
                        total_loss+=(xentropy)
                        
                        if i==2:
                            error = tf.argmax(logit,1)
                    
                            gt = tf.squeeze(tf.slice(current_caption_ind,[0,i],[tf.shape(data)[0],1]),1)
                
                            
                 word_embedding = tf.nn.embedding_lookup(embedding_map, tf.squeeze(tf.slice(current_caption_ind,[0,i],[tf.shape(data)[0],1]),1))
                 
                 image_embedding,att_weights = att.attention(data,state,int(data.shape[2]),Flags.ratio,True)

                 current_embedding = tf.concat([image_embedding, word_embedding],axis=-1)
     
    one_hot_path =  tf.nn.l2_normalize(one_hot_path,1)
                        
    probs_path = tf.nn.l2_normalize(probs_path,1)
    
    total_loss = tf.reduce_mean(total_loss)/(n_lstm_steps-2)         
    
    path_loss = tf.losses.cosine_distance(one_hot_path,probs_path,axis=1, loss_collection=None)
    
    total_loss += path_loss

    return total_loss,error,gt

def apply_lstm_gen(train_size,data,captions,weights,dim_hidden,dim_emb,dim_embed2,lstm,n_lstm_steps,n_words,wordtoix,embedding,class_inf,net):
    
    data = da.transform_data(data, is_training= False)
    
    data = apply_network_img(data, tf.shape(data)[0], is_training_net=False,
                             is_training_drop=False,net=net)
    
    mid = data.get_shape()[1]
    
    data = tf.reshape(data,[-1, mid*mid, int(data.shape[3])])
    
    state = get_initial_lstm(tf.reduce_mean(data,1),dim_hidden,False)

    current_caption_ind = tf.contrib.lookup.string_to_index(captions,wordtoix)
    
    embedding_map = tf.get_variable(
          name="map",
          shape=[n_words, dim_embed2],
          )


    size = tf.shape(current_caption_ind)
    
    word_embedding = tf.zeros([size[0], dim_embed2])
    
    total_loss =0.0
    probi = []
    all_words='#START#'
    with tf.variable_scope("RNN",reuse=tf.AUTO_REUSE):
        
        image_embedding,att_weights = att.attention(data,state,int(data.shape[2]),Flags.ratio,0)
   
        current_embedding = tf.concat([image_embedding, word_embedding],axis=-1)  
        
        for i in range(1,n_lstm_steps-1): 
                _, state = lstm(current_embedding, state)
                
                with tf.variable_scope("Caption",reuse=tf.AUTO_REUSE):
                    
#                    #perform a softmax classification to generate the next word in the caption
                        if class_inf==1:
                            logit = slim.fully_connected(state,n_words,activation_fn=None,normalizer_fn=None,scope='word_encoding')
                        else:
                            logit = slim.fully_connected(state,dim_embed2,activation_fn=None,normalizer_fn=None,scope='state_encoding')
                                                                            
                            context = slim.fully_connected(image_embedding,dim_embed2,activation_fn=None,normalizer_fn=None,scope='context')
                            
                            logit += context
                            
                            logit = tf.nn.tanh(logit)
                                                    
                            logit = slim.fully_connected(logit, n_words,activation_fn=None,normalizer_fn=None,scope='word_encoding')
#                
                best_word = tf.argmax(logit, 1)
                probi.append(tf.nn.softmax(logit))
                
                onehot = tf.one_hot(tf.squeeze(tf.slice(current_caption_ind,[0,i],[size[0],1]),1),n_words)
                
                if i==1:
                    one_hot_path = onehot
                    probs_path = tf.nn.softmax(logit)
                else:
                    one_hot_path += onehot
                    probs_path += tf.nn.softmax(logit)
                
                
                weight = tf.matmul(onehot,tf.cast(tf.expand_dims(weights,1),tf.float32),transpose_b=False)
                weight = train_size/weight
                        
                weight = tf.squeeze(weight/tf.reduce_mean(weight))
                                
                loss = tf.losses.softmax_cross_entropy(onehot,logits=logit, weights=weight,reduction=tf.losses.Reduction.NONE)
 
                total_loss += loss
 
                previous_word = tf.nn.embedding_lookup(embedding_map, best_word) 
                
                image_embedding,att_weights = att.attention(data,state,int(data.shape[2]),Flags.ratio,0)

                current_embedding = tf.concat([image_embedding, previous_word],axis=-1)
#                
                all_words += ' ' + tf.contrib.lookup.index_to_string(best_word,wordtoix)
            
                if i==2:
                    error = best_word
                    
                    gt = tf.squeeze(tf.slice(current_caption_ind,[0,i],[size[0],1]),1)
                     
    one_hot_path =  tf.nn.l2_normalize(one_hot_path,1)
                        
    probs_path = tf.nn.l2_normalize(probs_path,1)
    
    total_loss = tf.reduce_mean(total_loss)/(n_lstm_steps-2)         
    
    path_loss = tf.losses.cosine_distance(one_hot_path,probs_path,axis=1)
    
    total_loss += path_loss

    all_words += ' ' + '.'          
    return total_loss,error,all_words,gt

def change_checkpoint_name(var):
    return var.op.name.replace("train/densenet161",'densenet161')


def change_checkpoint_name3(var):
    return var.op.name.replace("train/vgg_16",'vgg_16')
                
def main(_):
    dim_embed1 = Flags.feature_maps# Number of feature maps
    dim_embed2 = Flags.word_embedding# Word Embedding
    dim_hidden = Flags.dim_hidden# Hidden dimension LSTM
    
    #### Built Vocab ####
    anno = da.get_captions(Flags.tfrecord_train)

    anno_val = da.get_captions(Flags.tfrecord_val)

    co_occur, vocab2,vocab, counts = da.preProBuildWordVocab(anno, word_count_threshold=20)
    
    if not os.path.exists('model/'):
        os.mkdir('model/')

    if not os.path.exists(Flags.train_dir_log):
        os.mkdir(Flags.train_dir_log)
##    
    np.save('model/co_occur', co_occur)
    
    np.save('model/vocab', vocab)
    
    np.save('model/counts', counts)
    
    word_embedding = da.loadGloVe(co_occur,dim_embed2,vocab2,1)

    n_words = len(vocab)
    maxlen = np.max( [x for x in map(lambda x: len(x.decode().split(' ')), anno) ] )    
    weights_vector = counts #2000
    
    g_1 = tf.Graph()
    with g_1.as_default():
        lstm = tf.nn.rnn_cell.LSTMCell(dim_hidden,state_is_tuple = False)
 
        lstm = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = 0.5,input_size = dim_embed1+dim_embed2,variational_recurrent=True, dtype=tf.float32)
                
        dataset_train = tf.data.TFRecordDataset(Flags.tfrecord_train, num_parallel_reads=4)

        dataset_val = tf.data.TFRecordDataset(Flags.tfrecord_val, num_parallel_reads=4)

        dataset_train = da.read_and_decode(dataset_train, Flags.train_batch_size, 1, len(anno))

        dataset_val = da.read_and_decode(dataset_val, Flags.val_batch_size, 0, len(anno_val))

        train_val_iterator = tf.data.Iterator.from_structure(dataset_train.output_types, dataset_train.output_shapes)
        batch_x, batch_y = train_val_iterator.get_next()

        train_iterator = train_val_iterator.make_initializer(dataset_train)
        val_iterator = train_val_iterator.make_initializer(dataset_val)
        
        with tf.variable_scope('train', reuse=tf.AUTO_REUSE):

            text = tf.string_split(batch_y)
            text = tf.sparse_tensor_to_dense(text, default_value=' ')
            
            total_loss,pred,gt_train = apply_lstm(len(anno),batch_x,text,weights_vector,dim_hidden,dim_embed1,dim_embed2,lstm,
                                    maxlen,n_words,vocab,word_embedding,Flags.class_inf,Flags.net)
            
            total_loss_val,pred_val,word,gt_val = apply_lstm_gen(len(anno),batch_x,text,weights_vector,dim_hidden,dim_embed1,dim_embed2,lstm,
                                            maxlen,n_words,vocab,word_embedding,Flags.class_inf,Flags.net)
        
        
        global_step = tf.Variable(0, trainable=False)
                   
        boundaries = [int(0.5 * Flags.how_many_training_steps * (len(anno) / (Flags.train_batch_size))),
                      int(0.75 * Flags.how_many_training_steps * (len(anno) / (Flags.train_batch_size)))]

        lr_init = [Flags.lr_init, Flags.lr_init/10, Flags.lr_init/100]

        lr = tf.compat.v1.train.piecewise_constant(global_step, boundaries, lr_init)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        reg_loss = tf.losses.get_regularization_loss()

        loss = total_loss + reg_loss

        train_op = optimizer.minimize(loss, global_step=global_step)

        vl = slim.get_variables_to_restore()

        saver = tf.train.Saver(var_list=vl, max_to_keep=Flags.how_many_training_steps)
 
        def init_points(sess):

            if Flags.net == 'D':
                restore = slim.get_model_variables('train/densenet161')

                restore = {change_checkpoint_name(var): var for var in restore}

                init_points_dense_op, init_points_dense_feed_dict = slim.assign_from_checkpoint(
                    os.path.join(Flags.checkpoint_dir, 'tf-densenet161.ckpt'), restore)

                sess.run(init_points_dense_op, init_points_dense_feed_dict)
            else:
                restore = slim.get_model_variables('train/vgg_16')

                restore = {change_checkpoint_name3(var): var for var in restore}

                init_points_vgg_op, init_points_vgg_feed_dict = slim.assign_from_checkpoint(
                    os.path.join(Flags.checkpoint_dir, 'vgg_16.ckpt'), restore)

                sess.run(init_points_vgg_op, init_points_vgg_feed_dict)
        
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer())
            
            sess.run(tf.local_variables_initializer())

            init_points(sess)
            
            tf.tables_initializer().run()
            
            train_writer = tf.summary.FileWriter(Flags.train_dir, sess.graph)

            validation_writer = tf.summary.FileWriter(Flags.val_dir, sess.graph)
            
            for k in range(Flags.fine_tune,Flags.how_many_training_steps):
                sess.run(train_iterator)
                scores = np.array([])
                error = np.array([])
                true_label = np.array([]) 

                steps = 0
                with slim.queues.QueueRunners(sess):
                        try:
                            with tqdm(total=len(anno)) as pbar:
                                while True:

                                    _,final_loss,err,gt = sess.run([train_op,total_loss,pred,gt_train])
                                    
                                    scores = np.append(scores, final_loss)
    
                                    error = np.append(error, err)
    
                                    true_label = np.append(true_label, gt)
                                    
                                    pbar.update(Flags.train_batch_size)
    
                                    print('Epoch %s /%s Step %s /%s: Batch_loss is %f' % (
                                    k, Flags.how_many_training_steps-1, steps,
                                    (len(anno) // Flags.train_batch_size), final_loss))   
                                    
                                    steps += 1
                                
                        except tf.errors.OutOfRangeError:
                            saver.save(sess, Flags.train_dir_log + '/model', global_step=k)
                    
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag='losses/Train_Loss', simple_value=np.mean(scores))])

                train_writer.add_summary(summary, k)

                summary = tf.Summary(value=[tf.Summary.Value(tag='BACC/Train_BACC',
                                                             simple_value=metrics.balanced_accuracy_score(
                                                                 true_label, error))])

                train_writer.add_summary(summary, k)

                summary = tf.Summary(value=[tf.Summary.Value(tag='Accuracy/Train_ACC',
                                                             simple_value=metrics.accuracy_score(true_label,
                                                                                                 error))])
                train_writer.add_summary(summary, k)

                print('Finished Training. BACC %f and Accuracy %f' % (
                    metrics.balanced_accuracy_score(true_label, error),
                    metrics.accuracy_score(true_label, error)))

                sess.run(val_iterator)

                scores = np.array([])
                error = np.array([])
                true_label = np.array([])

                with slim.queues.QueueRunners(sess):
                    try:
                        while True:
                            val_loss, err, gt = sess.run([total_loss_val, pred_val, gt_val])

                            scores = np.append(scores, val_loss)

                            error = np.append(error, err)

                            true_label = np.append(true_label, gt)

                    except tf.errors.OutOfRangeError:
                        pass

                    summary = tf.Summary(
                        value=[tf.Summary.Value(tag='losses/Val_Loss', simple_value=np.mean(scores))])

                    validation_writer.add_summary(summary, k)

                    summary = tf.Summary(value=[tf.Summary.Value(tag='BACC/Val_BACC',
                                                                 simple_value=metrics.balanced_accuracy_score(
                                                                     true_label, error))])

                    validation_writer.add_summary(summary, k)

                    summary = tf.Summary(value=[tf.Summary.Value(tag='Accuracy/Val_ACC',
                                                                 simple_value=metrics.accuracy_score(true_label,
                                                                                                     error))])

                    validation_writer.add_summary(summary, k)

                    print('Finished validation. BACC %f and Accuracy %f' % (
                        metrics.balanced_accuracy_score(true_label, error),
                        metrics.accuracy_score(true_label, error)))

            sess.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--tfrecord_train',
      type=str,
      default='data/Fold_1_T3/Training/train_full_norm.tfrecords',
      help='Path to folders of train labeled images.'
  )
  parser.add_argument(
      '--tfrecord_val',
      type=str,
      default='data/Fold_1_T3/Validation/val_full_norm.tfrecords',
      help='Path to folders of validation labeled images.'
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='model/training_fold1',
      help='Place to save summaries and checkpoints.'
  )
  parser.add_argument(
      '--train_dir_log',
      type=str,
      default='model/checkpoints_fold1',
      help='Place to save temporary checkpoints.'
  )
  parser.add_argument(
      '--val_dir',
      type=str,
      default='model/validation_fold1',
      help='Place to save summaries and checkpoints.'
     
  )
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
     default='checkpoints/',
     help='Path to checkpoint.'
  )
  parser.add_argument(
      '--net',
      type=str,
      default='V',
      help='Image encoder - "D" (densenet-161) or "V" (vgg16).'
  )
  parser.add_argument(
      '--feature_maps',
      type=int,
      default= 512,
      help='Number of feature maps: 2208 (DenseNet) or 512 (VGG).'
  )
  parser.add_argument(
      '--word_embedding',
      type=int,
      default= 50,
      help='Size of the word embedding.'
  )
  parser.add_argument(
      '--dim_hidden',
      type=int,
      default= 512,
      help='Hidden  LSTM.'
  )
  parser.add_argument(
      '--class_inf',
      type=int,
      default= 1,
      help='Label inference - state (1), state+context (2).'
  )
  parser.add_argument(
      '--fine_tune',
      type=int,
      default= 0,
      help='Train from sratch or fine-tune from previous version.'
  )
  parser.add_argument(
      '--lr_init',
      type=int,
      default=0.000001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=20,
      help='Size of your batch.'
  )
  parser.add_argument(
      '--val_batch_size',
      type=int,
      default=20,
      help='Size of your batch.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=120,
      help='How many epochs.'
  )
  parser.add_argument(
      '--ratio',
      type=int,
      default=None,
      help='Channel ratio reduction for channel attention ratio = int (if 0 without channel attention).'
  )
  Flags, unparsed = parser.parse_known_args()
  tf.app.run(main=main)