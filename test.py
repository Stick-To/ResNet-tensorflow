import ResNet as net
from dataprovider import dataprovider

train_set = [".\\tfrecord\\train.tfrecord"]
val_set = [".\\tfrecord\\val.tfrecord"]
trainset = dataprovider(train_set,3,1037,[90,160,3],1037,2)
valset = dataprovider(val_set,3,111,[90,160,3],111,2)

init_conv_param = {
    'kernel_size':7,
    'filters':64,
    'strides':2
}
init_pool_param = {
    'pool_size':3,
    'strides':2
}
#init_conv—init_pool—[64 64 64,128 128 128,256 256 256]—averge_pool——dense_layer--->[3,4,3]
testnet = net.ResNet(trainset,valset,0.01,1e-4,400,init_conv_param,init_pool_param,[3,4,3],'channels_last',False)
testnet.train_all_epochs()
