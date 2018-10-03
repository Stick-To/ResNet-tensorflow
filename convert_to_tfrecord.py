from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import os
import json
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def convert_to_tfrecord(imgdir, json_file, filename ,num_classes, newshape=(160,90)):
    f = open(json_file)
    decoder = json.load(f)
    tf.reset_default_graph()
    print('Writing', filename)
    tensor = tf.placeholder(tf.uint8)
    encode_jpeg = tf.image.encode_jpeg(tensor)
    session = tf.Session()
    with tf.python_io.TFRecordWriter(filename) as writer:
        i=0
        for entry in decoder:
            img_name = os.path.join(imgdir, entry['image_id'])
            print('Writing img',img_name,'now')
            label_ = int(entry['disease_class'])
            label = np.zeros(num_classes,dtype=np.int64)
            label[label_] = 1
            print(label)
            i+=1
            print('img num',i)
            label = label.tostring()
            image_raw = cv2.imdecode(np.fromfile(img_name,dtype=np.uint8),-1)
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



