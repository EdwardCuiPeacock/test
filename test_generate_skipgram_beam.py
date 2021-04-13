#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 13:27:04 2021

@author: edwardcui
"""
import pprint
import tempfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import skipgrams

import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
import apache_beam as beam


def make_preproc_func(vocabulary_size, window_size, negative_samples, feature_names):
    """Returns a preprocessing_fn to make skipgrams given the parameters."""
    def _make_skipgrams(s):
        """Numpy function to make skipgrams."""
        samples_out = []
        
        for i in range(s.shape[0]):
            pairs, labels = skipgrams(
                    s[i, :], vocabulary_size=100, window_size=window_size, 
                    negative_samples=negative_samples, seed=42,
                )
            samples = np.concatenate([np.atleast_2d(np.asarray(pairs)), np.asarray(labels)[:, None]], axis=1)
        
            samples_out.append(samples)
            
        samples_out = np.concatenate(samples_out, axis=0)
        return samples_out
    
    @tf.function
    def _tf_make_skipgrams(s):
        """tf nump / function wrapper."""
        y = tf.numpy_function(_make_skipgrams, [s], tf.int64)
        y.set_shape([None, 3])
        return y
    
    def _fn(inputs):
        """Preprocess input columns into transformed columns."""
        S = tf.stack([inputs[fname] for fname in feature_names], axis=1) # tf tensor
            
        out = _tf_make_skipgrams(S)
        
        output = {}
        output["target"] = out[:, 0]
        output["context"] = out[:, 1]
        output["label"] = out[:, 2]

        return output
    
    return _fn


def generate_skipgrams(data_uri, feature_names, vocabulary_size=10, window_size=2, negative_samples=0., save_path="temp"):
    def parse_tensor_f(x):
        xp = tf.io.parse_tensor(x, tf.int64)
        xp.set_shape([None])
        return {fname: xp[i] for i, fname in enumerate(feature_names)}
    
    raw_data = tf.data.TFRecordDataset(data_uri).map(parse_tensor_f).as_numpy_iterator()
    raw_data_schema = dataset_metadata.DatasetMetadata(
        schema_utils.schema_from_feature_spec(
            {fname: tf.io.FixedLenFeature([], tf.int64) for fname in feature_names}
        )
    )
    dataset = (raw_data, raw_data_schema)
    
    # Make the preprocessing_fn
    preprocessing_fn = make_preproc_func(vocabulary_size, window_size, negative_samples, feature_names)
    
    # Run the beam pipeline
    with beam.Pipeline() as pipeline:
        with tft_beam.Context(temp_dir=tempfile.mkdtemp(), desired_batch_size=2):
            transformed_dataset, transform_fn = (
            dataset | "Make Skipgrams" >> tft_beam.AnalyzeAndTransformDataset(preprocessing_fn)
            )
            print('Transformed dataset:\n{}'.format(pprint.pformat(transformed_dataset)))

            # pylint: disable=unused-variable
            transformed_data, transformed_metadata = transformed_dataset  
            saved_results = (transformed_data
                | "Write to TFRecord" >> beam.io.tfrecordio.WriteToTFRecord(
                    file_path_prefix=save_path, file_name_suffix=".tfrecords",
                    coder=tft.coders.example_proto_coder.ExampleProtoCoder(transformed_metadata.schema))
                )
            print('\nRaw data:\n{}\n'.format(pprint.pformat(raw_data)))
            print('Transformed data:\n{}'.format(pprint.pformat(transformed_data)))
            # Return the list of paths of tfrecords
            num_rows_saved = len(transformed_data)

    return saved_results, num_rows_saved
    
if __name__ == '__main__':
    features = tf.constant([[1,2, 3], [4, 5, 6], [1, 2, 3], [6, 7, 8], [2, 3, 5], [3, 5, 7], [1, 1, 1]], dtype="int64")
    
    data_uri = "./temp_feature_record.tfrecord"
    ds = tf.data.Dataset.from_tensor_slices(features).map(tf.io.serialize_tensor)
    writer = tf.data.experimental.TFRecordWriter(data_uri)
    writer.write(ds)
    
    saved_results, n = generate_skipgrams(data_uri, feature_names={f"s{i}" for i in range(3)})