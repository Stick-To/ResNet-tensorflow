from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
#ResNet-v2
class ResNet:
    def __init__(self,data_shape,num_classes,init_conv_param,init_pool_param,
                 blocks_list,data_format,is_bottleneck):

        self.data_shape = data_shape
        self.num_classes = num_classes
        self.init_conv_param = init_conv_param
        self.init_pool_param = init_pool_param
        self.blocks_list = blocks_list
        self.data_format = data_format
        self.is_training = True
        self.is_bottleneck = is_bottleneck
        self.filters = [self.init_conv_param['filters']*(2**i) for i in range(len(blocks_list))]
        self.global_step = tf.train.get_or_create_global_step()

        self._define_inputs()
        self._build_graph()
        self._init_session()
    def _define_inputs(self):
        shape = [None]
        shape.extend(self.data_shape)
        self.images = tf.placeholder(dtype=tf.float32,shape=shape,name='images')
        self.labels = tf.placeholder(dtype=tf.int32,shape=[None,self.num_classes],name='labels')
        self.learning_rate = tf.placeholder(dtype=tf.float32,shape=[],name='lr')
        self.l2_rate = tf.placeholder(dtype=tf.float32,shape=[],name='l2_rate')
        self.prob = tf.placeholder(dtype=tf.float32,shape=[],name='prob')
    def conv_layer(self, bottom, kernel_size, filters, strides=1, name='conv'):
        return tf.layers.conv2d(
            inputs=bottom,
            filters=filters,
            kernel_size = [kernel_size,kernel_size],
            strides=[strides,strides],
            padding='same',
            name = name,
            data_format = self.data_format
        )
    def max_pool(self, bottom, pool_size=2, strides=2):
        return tf.layers.max_pooling2d(bottom,
                [pool_size,pool_size],
                [strides,strides],
                data_format=self.data_format,
                padding='same'
        )
    def batch_norm(self,bottom):
        axes = 3 if self.data_format=='channels_last' else 1
        return tf.layers.batch_normalization(bottom,axis=axes,
                    training = self.is_training)
    def project_shortcut(self, bottom, filters, is_shutcut_pooling=True):
        strides = 2 if is_shutcut_pooling else 1
        return self.conv_layer(bottom, 1, filters, strides, name="project")
    def _build_block(self, bottom, filters, scope, is_project_shortcut=False):
        with tf.variable_scope(scope):
            if is_project_shortcut:
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
    def _build_bottleneck(self,bottom,filters,scope,is_project_shutcut=True,is_shutcut_pooling=False):
        with tf.variable_scope(scope):
            if is_project_shutcut:
                shortcut = self.project_shortcut(bottom,filters*4,is_shutcut_pooling)
            else:
                shortcut = bottom
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
    def stack_bottleneck(self, bottom, filters, scope, num_blocks,is_shutcut_pooling=True):
        input = bottom
        input = self._build_bottleneck(input, filters, scope+str(1),True,is_shutcut_pooling)

        for i in range(1, num_blocks):
            input = self._build_bottleneck(input, filters, scope+str(i+1),False, False)
        return input
    def _build_graph(self):
        self.init_conv = self.conv_layer(self.images,
                                         self.init_conv_param['kernel_size'],
                                         self.init_conv_param['filters'],
                                         strides=self.init_conv_param['strides'],
                                         name = 'init_conv')
        self.init_pool = self.max_pool(self.init_conv,self.init_pool_param['pooling_size'],
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

        axes = [1,2] if self.data_format=='channels_last' else [2,3]

        with tf.variable_scope('final_layers'):
            self.final_average_pool = tf.reduce_mean(residual_block,
                                                 axes,
                                                 keepdims=False,
                                                 name='final_average_pool'
            )
            self.final_dense =tf.layers.dense(self.final_average_pool,
                                              self.num_classes)
        with tf.variable_scope('result'):
            self.logits = tf.nn.softmax(self.final_dense)
            self.softmax_loss = tf.losses.softmax_cross_entropy(self.labels,self.final_dense,reduction=tf.losses.Reduction.MEAN)
            self.l2_loss = self.l2_rate * tf.add_n(
                        [tf.nn.l2_loss(var) for var in tf.trainable_variables()])
            self.loss = self.softmax_loss + self.l2_loss
            self.pred = tf.argmax(self.logits, 1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.labels,1),self.pred)
                                                   ,dtype=tf.float32))

        with tf.variable_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                    global_step=self.global_step)
    def _init_session(self):
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
    def train_one_epoch(self, images, labels, lr, l2_rate, prob):
        self.is_training = True
        loss,acc, _ = self.sess.run([self.loss,self.accuracy,self.optimizer],
                                    feed_dict={
                                        self.images:images,
                                        self.labels:labels,
                                        self.learning_rate:lr,
                                        self.l2_rate:l2_rate,
                                        self.prob:prob
                                    })
        return loss, acc
    def validate_one_epoch(self, images, labels, l2_rate, prob):
        self.is_training = False
        loss, acc = self.sess.run([self.loss,self.accuracy],
                                    feed_dict={
                                        self.images:images,
                                        self.labels:labels,
                                        self.learning_rate:0.,
                                        self.l2_rate:l2_rate,
                                        self.prob:prob
                                    })
        return loss, acc




