from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf

from model_lstm import AudioEventDetector
from audioset import vggish_input, vggish_params, vggish_postprocess, vggish_slim
# aed = AudioEventDetector()

import speech_recognition as sr
import os

class AudioRec(object):

    def __init__(self):
        self.r = sr.Recognizer()
        self.src = sr.Microphone()
        with self.src as source:
            print("Calibrating microphone...")
            self.r.adjust_for_ambient_noise(source, duration=2)


    def listen(self, save_path):
        with self.src as source:
            print("say something")
            # record for a maximum of 10s
            audio = self.r.listen(source, phrase_time_limit=10)
        # write audio to a WAV file
        with open(save_path, "wb") as f:
            f.write(audio.get_wav_data())


class VGGish(object):
    # Paths to downloaded VGGish files.
    checkpoint_path = 'vggish_model.ckpt'
    pca_params_path = 'vggish_pca_params.npz'

    def __init__(self):
        with tf.Graph().as_default():
            self.sess = tf.Session()
            vggish_slim.define_vggish_slim()
            vggish_slim.load_vggish_slim_checkpoint(self.sess, self.checkpoint_path)


    def extract_features(self, wav_path):
        # Produce a batch of log mel spectrogram examples.
        input_batch = vggish_input.wavfile_to_examples(wav_path)
        if input_batch.shape[0] < 1:
            print('{}: Audio sample shorter than 1 second. Ignoring ...', os.path.basename(wav_path))
            return None

        features_tensor = self.sess.graph.get_tensor_by_name(
                vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = self.sess.graph.get_tensor_by_name(
                vggish_params.OUTPUT_TENSOR_NAME)
        [embedding_batch] = self.sess.run([embedding_tensor],
                                    feed_dict={features_tensor: input_batch})
        # Postprocess the results to produce whitened quantized embeddings.
        pproc = vggish_postprocess.Postprocessor(self.pca_params_path)
        postprocessed_batch = pproc.postprocess(embedding_batch)
        return (np.float32(postprocessed_batch) - 128.) / 128.


    def __del__(self):
        print ("Closing the session..")
        self.sess.close()



if __name__ == '__main__':
    vggish = VGGish()
    ar = AudioRec()
    ar.listen("microphone-results.wav")
    features = vggish.extract_features("microphone-results.wav")

    if features is not None:
        print (features.shape)
        print (features)

    del vggish

