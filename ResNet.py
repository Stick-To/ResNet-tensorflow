from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#ResNet-v2
class ResNet:
    def __init__(self,dataprovider_train,dataprovider_test,init_learning_rate,epochs,init_conv_param,init_pool_param,blocks_list,data_format,is_bottleneck):

        """
            dataprovider_train:class dataprovider,trainset inside
            dataprovider_test:class dataprovier, if not None test it every epoch during train process
                                        and  default be used  in test() function
            init_learning_rate:the learning_rate when train process start
            epochs:epochs
            init_conv_param:dict, the parameters of init_conv  inside
            init_pool_param:dict, the parameters of init_pool  inside
            block_list:list,the times of every filters(the filters parameter of conv2d in tf) to be use,the first block use the
                            filters of init_conv and double it after it is used the times that this list claim
            data_format:'channels_first' or 'channels_last'
            is_bottleneck:Whether use bottleneck block
        """

        self.dataprovider_train = dataprovider_train
        self.data_shape = dataprovider_train.data_shape
        self.num_classes = dataprovider_train.num_classes
        self.train_initop = dataprovider_train.initop
        self.train_iterator = dataprovider_train.iterator
        self.num_train = dataprovider_train.num_samples
        self.train_batch_size = dataprovider_train.batch_size
        self.train_labels,_ ,self.train_images = self.train_iterator.get_next()
        self.train_images =tf.cast(self.train_images,tf.float32)

        self.is_test = True if dataprovider_test is not None else False
        if self.is_test:
            self.dataprovider_test = dataprovider_test
            self.test_initop = dataprovider_test.initop
            self.test_iterator = dataprovider_test.iterator
            self.num_test = dataprovider_test.num_samples
            self.test_batch_size = dataprovider_test.batch_size
            self.test_labels,_ ,self.test_images = self.test_iterator.get_next()
            self.test_images =tf.cast(self.test_images,tf.float32)



        self.learning_rate = init_learning_rate


        self.epochs = epochs
        self.init_conv_param = init_conv_param
        self.init_pool_param = init_pool_param
        self.blocks_list = blocks_list
        self.data_format = data_format
        self.is_training = True
        self.is_bottleneck = is_bottleneck
        self.filters = [self.init_conv_param['filters']*(2**i) for i in range(len(blocks_list))]

        self.global_step = tf.get_variable(name='global_step',initializer=tf.constant(0,dtype=tf.int32),
                                           trainable=False)

        self._define_inputs()
        self._build_graph()
        self._init_session()
    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(dtype=tf.float32,shape=shape,name='images')
        self.labels = tf.placeholder(dtype=tf.int32,shape=[None,self.num_classes],name='labels')
    def conv_layer(self, bottom, kernel_size, filters, strides=1, name=''):
        #pad = kernel_size // 2
        #if self.data_format == 'channels_last':
        #    bottom_pad = tf.pad(bottom, [[0,0],[pad,pad],[pad,pad],[0,0]])
        #else:
        #    bottom_pad = tf.pad(bottom, [[0,0],[0,0],[pad,pad],[pad,pad]])
        return tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size = [kernel_size,kernel_size],
            strides=[strides,strides],
            padding='SAME',
            use_bias = True,
            name = name,
            data_format = self.data_format
        )
    def max_pool(self, bottom, pool_size=2, strides=2):
        return tf.layers.max_pooling2d(bottom, [pool_size,pool_size], [strides,strides],data_format=self.data_format,padding='SAME')
    def batch_norm(self,bottom):
        return tf.layers.batch_normalization(bottom,axis=3 if self.data_format=='channels_last' else 1,
                    training = self.is_training)
    def project_shortcut(self, bottom, filters, is_shutcut_pooling=True):
        strides = 2 if is_shutcut_pooling else 1
        return self.conv_layer(bottom, 1, filters, strides, name="project")
    def _build_block(self, bottom, filters, scope, use_project_shortcut=False):
        with tf.variable_scope(scope):
            if(use_project_shortcut==True):
                shortcut = self.project_shortcut(bottom,filters)
                strides = 2
            else:
                shortcut = bottom
                strides = 1
            batch_norm = self.batch_norm(bottom)
            relu = tf.nn.relu(batch_norm)
            conv = self.conv_layer(relu,3,filters,strides=strides,name='conv_1')
            batch_norm = self.batch_norm(conv)
            relu = tf.nn.relu(batch_norm)
            conv = self.conv_layer(relu,3,filters,name='conv_2')
            return shortcut + conv
    def _build_bottleneck(self,bottom,filters,scope,is_shutcut_pooling=False):
        with tf.variable_scope(scope):
            shortcut = self.project_shortcut(bottom,filters*4,is_shutcut_pooling)
            strides = 2 if is_shutcut_pooling else 1
            batch_norm = self.batch_norm(bottom)
            relu = tf.nn.relu(batch_norm)
            conv = self.conv_layer(relu,1,filters,name='conv_1')
            batch_norm = self.batch_norm(conv)
            relu = tf.nn.relu(batch_norm)
            conv = self.conv_layer(relu,3,filters,strides=strides,name='conv_2')
            batch_norm = self.batch_norm(conv)
            relu = tf.nn.relu(batch_norm)
            conv = self.conv_layer(relu,1,filters*4,name='conv_3')
            return conv+shortcut
    def stack_block(self, bottom, filters, scope,  num_blocks, use_project_shortcut):
        input = bottom
        input = self._build_block(input, filters, scope+str(1), use_project_shortcut)
        for i in range(1, num_blocks):
            input = self._build_block(input, filters, scope+str(i+1), False)
        return input
    def stack_bottleneck(self, bottom, filters, scope,  num_blocks,is_shutcut_pooling=True):
        input = bottom
        input = self._build_bottleneck(input, filters, scope+str(1),is_shutcut_pooling)
        for i in range(1, num_blocks):
            input = self._build_bottleneck(input, filters, scope+str(i+1),False)
        return input
    def _build_graph(self):
        self.init_conv = self.conv_layer(self.images,
                                         self.init_conv_param['kernel_size'],
                                         self.init_conv_param['filters'],
                                         strides=self.init_conv_param['strides'],
                                         name = 'init_conv')
        self.init_pool = self.max_pool(self.init_conv,self.init_pool_param['pool_size'],
                                       self.init_pool_param['strides'])

        if self.is_bottleneck:
            stack_block_fn = self.stack_bottleneck
            residual_block = stack_block_fn(self.init_pool, self.filters[0] ,'residual_blcok_1',self.blocks_list[0], False)
            for i in range(1, len(self.blocks_list)):
                residual_block = stack_block_fn(residual_block, self.filters[i] ,'residual_block_'+str(i+1),self.blocks_list[i], True)
        else:
            stack_block_fn = self.stack_block
            residual_block = stack_block_fn(self.init_pool, self.filters[0] ,'residual_blcok_1',self.blocks_list[0], False)
            for i in range(1, len(self.blocks_list)):
                residual_block = stack_block_fn(residual_block, self.filters[i] ,'residual_block_'+str(i+1),self.blocks_list[i], True)

        axes = [2,3] if self.data_format=='channels_last' else [1,2]

        with tf.variable_scope('final_layers'):
            #use tf.reduce_mean instead of average_pool like tensorflow.models.offcial.resnet
            self.final_average_pool = tf.reduce_mean(residual_block,
                                                 axes,
                                                 keepdims=False,
                                                 name='final_average_pool'
            )
            self.final_dense =tf.layers.dense(self.final_average_pool,
                                          self.num_classes)
        with tf.variable_scope('result'):
            self.logits = tf.nn.softmax(self.final_dense)
            self.loss = tf.losses.softmax_cross_entropy(self.labels,self.logits,reduction=tf.losses.Reduction.MEAN)
            self.pred = tf.argmax(self.logits, 1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.labels,1),self.pred)
                                                   ,dtype=tf.float32))

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                    global_step=self.global_step)
    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
    def train_one_epoch(self):
        total_loss = []
        total_acc = []
        self.sess.run(self.train_initop)
        for i in range(self.num_train // self.train_batch_size):
            images, labels = self.sess.run([self.train_images,self.train_labels])
            loss,acc, _ = self.sess.run([self.loss,self.accuracy,self.optimizer],
                                        feed_dict={
                                            self.images:images,
                                            self.labels:labels
                                        })
            total_loss.append(loss)
            total_acc.append(acc)
        mean_loss = np.mean(total_loss)
        mean_acc = np.mean(total_acc)
        return mean_loss, mean_acc

    def test(self):
        total_loss = []
        total_acc = []
        self.sess.run(self.test_initop)
        for i in range(self.num_test // self.test_batch_size):
            images, labels = self.sess.run([self.test_images,self.test_labels])
            loss,acc = self.sess.run([self.loss,self.accuracy],
                                     feed_dict={
                                         self.images:images,
                                         self.labels:labels
                                     })
            total_loss.append(loss)
            total_acc.append(acc)
        mean_loss = np.mean(total_loss)
        mean_acc = np.mean(total_acc)
        return mean_loss, mean_acc

    def train_all_epochs(self):

        for epoch in range(self.epochs):
            print('-'*25,'epoch: ',epoch,'-'*25)
            loss, acc = self.train_one_epoch()
            print('train opech: ',epoch,' mean loss: ',loss," mean acc: ",acc)
            if self.dataprovider_test is not None:
                loss, acc = self.test()
                print('val opech: ',epoch,' mean loss: ',loss," mean acc: ",acc)

