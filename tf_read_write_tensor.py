#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 20:38:59 2021

@author: edwardcui
"""


#%%

import tensorflow as tf


S = tf.constant([[1,2, 3], [4, 5, 6], [1, 2, 3], [6, 7, 8], [2, 3, 5], [3, 5, 7]], dtype="float32")

def tensor2tfrecord(S, data_uri="temp.tfrecord", compression_type="GZIP"):
    """Write a tensor to a single TFRecord file."""
    ds = tf.data.Dataset.from_tensor_slices(tf.cast(S, "float32")).map(tf.io.serialize_tensor)
    writer = tf.data.experimental.TFRecordWriter(data_uri, compression_type=compression_type)
    writer.write(ds)


def record2dataset(data_uri="temp.tfrecord", compression_type="GZIP"):
    """Read from a TFRecord file or a list of files."""
    def parse_tensor_f(x):
        xp = tf.io.parse_tensor(x, tf.float32)
        #xp = tf.reshape(xp, [-1, 3])
        xp.set_shape([None])
        #print(xp.shape)
        return (xp[0], xp[1]), xp[2]
        #return (tf.squeeze(xp[:, 0]), tf.squeeze(xp[:, 1])), tf.squeeze(xp[:, 2])
    
    dataset = tf.data.TFRecordDataset(data_uri, compression_type=compression_type
                                      ).map(parse_tensor_f)
    return dataset
    
tensor2tfrecord(S)

dataset = record2dataset().batch(2)

for i in dataset:
    print(i)
    
# %%

