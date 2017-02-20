#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:57:31 2017

@author: yliu.Edward@hadoop

This script is designed for the SH and SZ Stock Exchange to choose the best several shares
in different industies.
Base on the distributed LSTM RNN. 

"""

import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

FLAGS = tf.app.flags.FLAGS

#parameters setting
tf.app.flags.DEFINE_float('learning_rate', 0.0001, 'Default learning rate')
tf.app.flags.DEFINE_integer('step_to_validate', 1000, 'Interval of validating')
tf.app.flags.DEFINE_integer('LSTM_step', 10, "LSTM_Cell's time step")
tf.app.flags.DEFINE_integer('input_size', 40, 'input features')
tf.app.flags.DEFINE_integer('cell_size', 80, 'neurons in hidden layer')
tf.app.flags.DEFINE_integer('batch_size',10, 'the shares pool dividens')
tf.app.flags.DEFINE_integer('h1_size',100, 'the hidden layer neurons')
tf.app.flags.DEFINE_integre('num_size', 3, 'number of LSTM-cells for multi-LSTM')

#distributed settting
tf.app.flags.DEFINE_string('ps_hosts', '', 'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('worker_hosts', '', 'Comma-separated list of hostname:port pairs')
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
tf.app.flags.DEFINE_integer('issync', 0 , 'the model of training. 1 is sync, 0 is no-sync')

#hyperparameters setting
learning_rate = FLAGS.learning_rate
step_to_validate = FLAGS.step_to_validate
cell_size = FLAGS.cell_size
batch_size = FLAGS.batch_size
LSTM_step = FLAGS.LSTM_step
input_size = FLAGS.input_size
h1_size = FLAGS.h1_size
num_size = FLAGS.num_size

#distributed loading
ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts.split(",")
cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)








def main():


#==============================================================================
# --------main RNN structure--------
# Regression on LSTM-RNN
#==============================================================================
        

    class LSTMRNN():
        
        #initial setting
        def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, h1_size,num_size):
            self.n_steps = n_steps
            self.input_size = input_size
            self.output_size = output_size
            self.cell_size = cell_size
            self.batch_size = batch_size
            self.num_size = num_size
            self.h1_size = h1_size
    
            
            with tf.name_scope('inputs'):
                self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
                self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
                self.keep_prob = tf.placeholder(tf.float32, [1], name='keep_prob')
                
            with tf.name_scope('in_hidden'):
                self.add_input_layer()
                    
            with tf.name_scope('LSTM_Cell'):
                self.add_cell_layer(self.num_size,self.keep_prob)
                
            with tf.name_scope('hidden_1'):
                self.add_h1_layer()
                        
            with tf.name_scope('out_hidden'):
                self.add_output_layer()
        
            with tf.name_scope('cost'):
                self.compute_cost()
    
            with tf.name_scope('train'):
                self.train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
      
        def add_input_layer(self):
            l_in_x = tf.reshape(self.xs,[-1,self.input_size], name='x_input')
            Ws_in = tf.Variable(tf.truncated_normal([self.input_size, self.cell_size], mean=0.01, stddev=0.5))
            bs_in = tf.Variable(tf.zeros([self.cell_size,])+0.01)
            l_in_y = tf.nn.relu(tf.matmul(l_in_x,Ws_in)+bs_in)
            l_in_y = tf.reshape(l_in_y,[-1,self.n_steps,self.cell_size],name='cell_input')

        def add_cell_layer(self, num_size=self.num_size, keep_prob=self.keep_prob):
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True), output_keep_prob=keep_prob)
            lstm_cells = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*self.num_size, state_is_tuple=True)
            cells_init_state = lstm_cells.zero_state(self.batch_size,dtype=tf.float32)
            cells_outputs, cells_final_state = tf.nn.dynamic_rnn(lstm_cells, l_in_y, initial_state=cells_init_state, time_major=False)
        
        def add_h1_layer(self):
            h1_x = tf.reshape(cells_outputs, [-1,self.cell_size])
            Ws_h1 = tf.Variable(tf.truncated_normal([self.cell_size, self.h1_size], mean=0.01, stddev=0.5))
            bs_h1 = tf.Variable(tf.zeros([self.h1_size,])+0.01)
            h1_y = tf.nn.relu(tf.matmul(h1_x,Ws_h1)+bs_h1)            
                
        
        def add_output_layer(self):
            l_out_x = tf.reshape(h1_y,[-1,self.h1_size],name = 'y_input')
            Ws_out = tf.Variable(tf.truncated_normal([self.h1_size, self.output_size], mean=0.01, stddev=0.5))
            bs_out = tf.Variable(tf.zeros([self.output_size,])+0.01)
            pred = tf.matmul(l_out_x,Ws_out)+bs_out


        def compute_cost(self):
            losses = tf.nn.seq2seq.sequence_loss_by_example([tf.reshape(self.pred,[-1],name='reshape_pred')],
                                                [tf.reshape(self.ys,[-1],name='reshape_target')],
                                                [tf.ones([self.batch_size*self.n_steps],dtype='float32')],
                                                average_across_timesteps=True, 
                                                softmax_loss_function=ms_error, 
                                                name='losses')
            cost=tf.div(tf.reduce_sum(losses),self.batch_size)
            
    
        def ms_error(self, y_pre, y_target):
            return tf.squre(tf.subtract(y_pre,y_target))
   
    #traing setting
    issync = FLAGS.issync
    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == 'worker':
        with tf.device(tf.train.replica_device_setter(
                   worker_device='/job:worker/task:%d' % FLAGS.task_index,
                   cluster = cluster)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            model = LSTMRNN(LSTM_step, input_size, 1, cell_size, batch_size, h1_size,num_size)
        
            if issync == 1:
                print('syn-model is not supported temporarily!')
            #rep_op =tf.train.SyncReplicasOptimizer(model.train_op, replicas_to_aggregate=len(worker_hosts),replic_id=FLAGS.task_index,total_num_replicas=len(worker_hosts),use_locking=True)
            else:
                init_op = tf.global_variables_initializer()
            
                saver = tf.train.Saver()
                summary_op = tf.merge_all_summaries()
            
            sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), logdir='./logdir/', init_op=init_op, summary_op=None, saver=saver, global_step=global_step)
            with sv.prepare_or_wait_for_session(server.target) as sess:
                     
                #read data from csv
                sz_trade = pd.read_csv('/home/hadoop/PycharmProjects/finance/sz_trade.csv')        
                close = list(sz_trade['close'])
                close_compute = np.array(close[2:])
                close = np.array(close[:len(close)-2])
                return_list = (close_compute - close)/close
                sz_trade = sz_trade.iloc[2:,:]
                sz_trade['return'] = return_list

                data = np.array(sz_trade)
                data_x = data[:,1:5]
                data_y = data[:,5]

                data_x = MinMaxScaler().fit_transform(data_x)
                data_y = Normalizer().fit_transform(data_y)

                data_x= data_x.reshape([-1,5,4])
                data_y = data_y.reshape([-1,5,1])
                train_data_x = np.array(data_x[:250,:,:],dtype = 'float32')        
                train_data_y = np.array(data_y[:250,:,:],dtype = 'float32')
                vali_data_x = np.array(data_x[250:,:,:],dtype = 'float32')
                vali_data_y = np.array(data_y[250:,:,:],dtype = 'float32')
                
                epoch = 0
                max_epoch = 1000
                while epoch <= max_epoch:
                    batch_start = 0 
                    for batch_p in range(25): # range will depend on the train_data.rows
                        if batch_p ==0:
                            feed_dict_train = {model.xs:train_data_x[0:batch_size,:,:], model.ys:train_data_y[0:batch_size,:,:],model.keep_prob: 1}
                        else:
                            batch_start = batch_p*batch_size
                            batch_end = (batch_p+1)*batch_size
                            feed_dict_train = {model.xs:train_data_x[batch_start:batch_end,:,:], model.ys:train_data_y[batch_start:batch_end,:,:], model.cells_init_state:state, model.keep_prob: 0.7}
                        sess.run(model.train_op,feed_dict=feed_dict_train)
                        state = sess.run(model.cells_final_state,feed_dict=feed_dict_train)
                        cost= sess.run(tf.cast(model.cost,dtype=tf.float32),feed_dict = feed_dict_train)
                        mean_pred= sess.run(tf.reduce_mean(model.pred), feed_dict = feed_dict_train)
                        print(("epoch: %d, batch: %d, cost: %f, mean_pred: %f" %(epoch, batch_p+1, cost, mean_pred))) 
                        epoch += 1
                        loss_list = []
                        if epoch % 2 == 0:
                            print('------------------------------------------------')
                            batch_start = 0
                            for batch_p in range(9): # range will depend on the vali_data.rows
                                batch_start = batch_p*batch_size
                                batch_end = (batch_p+1)*batch_size
                                feed_dict_vali = {model.xs:vali_data_x[batch_start:batch_end,:,:], model.ys:vali_data_y[batch_start:batch_end,:,:], model.cells_init_state:state, model.keep_prob: 1}
                                state = sess.run(model.cells_final_state,feed_dict=feed_dict_vali)
                                vali_pred = sess.run(model.pred,feed_dict = feed_dict_vali)
                                loss_mean = sess.run(tf.reduce_mean(tf.abs(tf.reshape(vali_pred,[-1])/tf.reshape(feed_dict_vali[ys],[-1]))))
                                print("vali_epoch:{0}, batch: {1}, loss: {2}%".format(epoch-1,batch_p+1, loss_mean*100))
                                loss_list.append(loss_mean)
                        if np.mean(loss_list)<=0.4:
                            sv.saver.save(sess, './logdir/')
                            sv.stop()
                            print('Train process has completed!')
                sv.stop()
        

if __name__ == '__main__':
    tf.app.run()
