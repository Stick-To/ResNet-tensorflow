from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import glob
import cv2

#create tfrecord
#default:img_name[-5] is the label (number),convert to one_hot<----(zeors(3,np.int64),in this example)
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def convert_to_tfrecord(imgdir, filename ,newshape=(160,90)):
    image_list= glob.glob(imgdir+"*.jpg")
    tf.reset_default_graph()
    print('Writing', filename)
    tensor = tf.placeholder(tf.uint8)
    encode_jpeg = tf.image.encode_jpeg(tensor)
    session = tf.Session()
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in range(len(image_list)):
            img_name = image_list[index]
            print('Writing img',img_name,'now')
            label_ = int(img_name[-5])
            label = np.zeros(3,dtype=np.int64)
            label = label.tostring()
            image_raw = cv2.imread(img_name)
            image_raw = cv2.resize(image_raw, newshape)
            shape = np.array(image_raw.shape, np.int32).tobytes()
            jpeg_bytes = session.run(encode_jpeg, feed_dict={tensor: image_raw})
            example = tf.train.Example(features=tf.train.Features(
                    feature={
                            'label': _bytes_feature(label),
                            'shape': _bytes_feature(shape),
                            'image_raw':_bytes_feature(jpeg_bytes)
                     }))
            writer.write(example.SerializeToString())
    session.close()
    print('Writing', filename,'done')



