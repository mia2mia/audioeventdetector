# Audio Event Detection

### Installation

The code should be compatible with both Python 2 and Python 3 (unverified).

#### VGGish depends on the following Python packages:

* [`numpy`](http://www.numpy.org/)
* [`scipy`](http://www.scipy.org/)
* [`resampy`](http://resampy.readthedocs.io/en/latest/)
* [`tensorflow`](http://www.tensorflow.org/) - >= v1.8.0
* [`six`](https://pythonhosted.org/six/)

These are all easily installable via, e.g., `pip install numpy` (as in the
example command sequence below).

Any reasonably recent version of these packages should work. TensorFlow should
be at least version 1.0.  We have tested with Python 2.7.6 and 3.4.3 on an
Ubuntu-like system with NumPy v1.13.1, SciPy v0.19.1, resampy v0.1.5, TensorFlow
v1.2.1, and Six v1.10.0.

#### VGGish also requires downloading two data files into the audioset folder:

* [VGGish model checkpoint](https://storage.googleapis.com/audioset/vggish_model.ckpt),
  in TensorFlow checkpoint format.
* [Embedding PCA parameters](https://storage.googleapis.com/audioset/vggish_pca_params.npz),
  in NumPy compressed archive format.

#### Audio Recording functionality requires the following packages:

* [`SpeechRecognition`](https://pypi.org/project/SpeechRecognition/) - Install using pip
* [`pyAudio`](https://people.csail.mit.edu/hubert/pyaudio/) - See installation instructions for your platform

#### Training/Loading the classifier needs the following packages:

* [`keras`](https://keras.io/#installation) - >= v2.2.0
* [`Matplotlib`](https://matplotlib.org/users/installing.html)
* [`H5py`](https://www.h5py.org/)

#### Download the Dataset if you want to train:
The hdf5 data can be downloaded [here](https://drive.google.com/open?id=0B49XSFgf-0yVQk01eG92RHg4WTA).

### Usage
This section provides detailed information about the code structure and how to train and test your models from scratch.

#### Code Structure

* AUDIOSET - folder containing audioset utilities for extracting VGGish utilities
* DATASET - post-processed dataset for 10 general categories
* DATASET_FACT - post-processed dataset for 10 factory categories
* DATASET_HOSP - post-processed dataset for 10 hospital categories
* DATASET_GENERAL - post-processed dataset for 19 general categories
* PACKED_FEATURES - original dataset containing released VGGish embeddings for 527 categories with multi-hot labels
* WEIGHTS - folder containing the best trained models for all the above categories
* `config.py` - config file for 10 general categories
* `config_fact.py` - config file for 10 factory categories
* `config_hosp.py` - config file for 10 hospital categories
* `config_general.py` - config file for 19 general categories
* `class_labels_indices.csv` - CSV file containing class names to label index mappings
* `confusion_general.csv` - confusion matrix for original 21 general categories
* `confusion_hosp.csv` - confusion matrix for hospital categories
* `model_lstm.py` - the main architecture for Audio Event Detector. Contains a Bidirectional LSTM layer trained on top of audioset embeddings, followed by a Global Average Pooling Layer and a final Dense layer with Softmax activation.
* `model_dense.py` - Deprecated. Fully Connected Layers trained on top of embeddings with softmax activation.
* `model_conv.py` - Deprecated. CNN trained on top of VGGish embeddings followed by fully connected layers and softmax activation.
* `process_data.py` - Code required to extract required subset of data from original audioset and store it as `npy` files for use during training. 
* `realtime.py` - Code required for Audio event detection in real time. Needs correct config and path to correct pre-trained model. 
* `utils.py` - file containing plotting and directory creation utilities

#### How to test the pre-trained model

Open `realtime.py` and find the `#TODO:` line and load the correct config file under that. For example, for the factory category:
```python
import config_fact as config
```

Open the correct config file, for instance `config_fact.py` and change the `TRAINED_MODEL_PATH` variable to reflect the correct path to the trained model, for example `weights/aed_fact_model_0718_134223_0.8236.h5`

We're ready to start our realtime application. Simply run:
```sh
$ python realtime.py
```

#### How to train from scratch

1. Make sure the packed_features directory is in your root directory for the repo.
2. Make a list of the classes you want to train your audio detector for by going through the `class_labels_indices.csv`.
3. Open the config file `config.py` and update the `CLASSES` dictionary with the correct label indices and class names from `class_labels_indices.csv`. Change the path to the folder where the extracted data should be saved by modifying `X_TRAIN_PATH`, `Y_TRAIN_PATH` and so on.
3. Open `process_data.py` and load the correct config under the `#TODO` line. For example:
```python
import config_fact as config
```
4. We're ready to start extracting our dataset from the original data. Run:
```sh
$ python process_data.py
```
WARNING: This takes 10-15 mins to load such a big dataset.

5. Now your dataset should be ready for use in training in a folder such as `DATASET_FACT`. Open the config file again and set parameters for training your model such as `NUM_LSTM_LAYERS` and `LEARNING_RATE` etc.

6. We can start our model training by runnning:
```sh
$ python model_lstm.py
```
The best model from training will be saved under weights directory.
