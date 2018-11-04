
import tensorflow as tf
import keras
from keras.datasets import cifar10
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import ResNet as net

device_name = tf.test.gpu_device_name()
if device_name is not '':
    print('Found GPU Device!')
else:
    print('Found GPU Device Failed!')


data_shape = (32,32,3)
num_train = 50000
num_test = 10000
num_classes = 10
batch_size = 32
epochs = 200

(x_train, y_train),(x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=30.
).flow(x_train, y_train, batch_size=batch_size)
test_datagen = ImageDataGenerator().flow(x_test, y_test, batch_size=batch_size)


init_conv_param = {
    'kernel_size':7,
    'filters':64,
    'strides':2
}
init_pool_param = {
    'pooling_size':3,
    'strides':2
}

lr = 0.001
l2_rate = 1e-4
drop_prob = 0.3
reduce_lr_epoch = [15]
testnet = net.ResNet((32,32,3),10,init_conv_param,init_pool_param,[3,4,6,3],'channels_last',True)
for epoch in range(epochs):
    print('-'*50,'epoch',epoch,'-'*50)
    if(epoch in reduce_lr_epoch):
        lr /= 10
        print('learning rate reduced to',lr)
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    for iter in range(num_train//batch_size):
        images, labels = train_datagen.next()
        loss, acc = testnet.train_one_epoch(images, labels, lr, l2_rate, drop_prob)
        sys.stdout.write('\r>> train iter '+str(iter)+' loss '+str(loss)+' acc '+str(acc))
        train_acc.append(acc)
        train_loss.append(loss)
    train_mean_loss = np.mean(train_loss)
    train_mean_acc = np.mean(train_acc)
    sys.stdout.write('\n')
    print('>> train mean loss',train_mean_loss,' train mean acc:',train_mean_acc)
    for iter in range(num_test//batch_size):
        images, labels = test_datagen.next()
        loss, acc = testnet.validate_one_epoch(images, labels, l2_rate, drop_prob)
        sys.stdout.write('\r>> test iter '+str(iter)+' loss '+str(loss)+' acc '+str(acc))
        val_acc.append(acc)
        val_loss.append(loss)
    val_mean_loss = np.mean(val_loss)
    val_mean_acc = np.mean(val_acc)
    sys.stdout.write('\n')
    print('>> test mean loss',val_mean_loss,' test mean acc:',val_mean_acc)

