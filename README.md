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
