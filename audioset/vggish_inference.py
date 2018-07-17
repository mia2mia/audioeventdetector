# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A smoke test for VGGish.

This is a simple smoke test of a local install of VGGish and its associated
downloaded files. We create a synthetic sound, extract log mel spectrogram
features, run them through VGGish, post-process the embedding ouputs, and
check some simple statistics of the results, allowing for variations that
might occur due to platform/version differences in the libraries we use.

Usage:
- Download the VGGish checkpoint and PCA parameters into the same directory as
    the VGGish source code. If you keep them elsewhere, update the checkpoint_path
    and pca_params_path variables below.
- Run:
    $ python vggish_smoke_test.py
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

import time
import os

print('\nTesting your install of VGGish\n')

# Paths to downloaded VGGish files.
checkpoint_path = 'vggish_model.ckpt'
pca_params_path = 'vggish_pca_params.npz'

def extract_vggish_features(wav_path):
    # Produce a batch of log mel spectrogram examples.
    input_batch = vggish_input.wavfile_to_examples(wav_path)
    if input_batch.shape[0] < 1:
        print('{}: Audio sample shorter than 1 second. Ignoring ...', os.path.basename(wav_path))
        return None

    # print('Log Mel Spectrogram example: ', input_batch[0])

    # Define VGGish, load the checkpoint, and run the batch through the model to
    # produce embeddings.
    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim()
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)

        features_tensor = sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)
        [embedding_batch] = sess.run([embedding_tensor],
                                    feed_dict={features_tensor: input_batch})
    # Postprocess the results to produce whitened quantized embeddings.
    pproc = vggish_postprocess.Postprocessor(pca_params_path)
    postprocessed_batch = pproc.postprocess(embedding_batch)
    return postprocessed_batch
