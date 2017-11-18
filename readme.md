Program Abstraction Classifier
===================================
Given a set of input to a program or function, the model should learn and output either valid (`1`) or invalid (`0`).
Invalid inputs will result in a program crash.

The repository is publicly available at https://github.com/yulonglong/ProgramAbstractionClassifier 

- Steven Kester Yuwono (sky@u.nus.edu)
- Thanh Toan Nguyen (toannt@comp.nus.edu.sg) 

**Requirements and Environment:**  

Please note that it has only been tested using:
- Numpy 1.12.0
- Keras 1.1.1
- Theano 0.8.2
- Pydot 1.1.0
- Nltk 3.2.2
- Python 2.7.12
- Ubuntu 16.04

To install the aforementioned Python libraries, simply run `setup.sh` in the current directory (i.e., `$> ./setup.sh`)
If you want to install manually:
- `pip install numpy==1.12.0`
- `pip install theano==0.8.2`
- `pip install keras==1.1.1`
- `pip install nltk==3.2.2`
- `pip install pydot==1.1.0`
- `pip install h5py==2.6.0`
- `pip install matplotlib==1.5.3`


**Dataset preparation:**

To download and prepare the dataset, simply run `refresh_dataset.sh` in the current directory (i.e., `$> ./setup.sh`). 
Dataset has been prepared in another GitHub repository (https://github.com/thanhtoantnt/precond-inference).
Some of the datasets (Polynomial, String.set, List.nth) are very large (more than 300MB each), we are not able to store them in GitHub. Please let us know if you need those files.


**Training:**

To train the model, read the shell script `run_train_nn.sh`
It contains all the necessary information to run the model using `Complex` Dataset.
To train the model, run `run_train_nn.sh` with two arguments.
i.e, `$> ./run_train_nn.sh 0 TITANX` where 0 is the GPU number and TITANX is the GPU name.

The results of the run will be saved in local directory with name starting with `expt`.

