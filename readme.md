Program Verification Classifier
===================================
Given a set of input to a program or function, the model should learn and output either valid (1) or invalid (0).
Invalid inputs will result in a program crash.

The repository is publicly available at https://github.com/yulonglong/ProgramAbstractionClassifier  

**Requirements and Environment:**  
- Ubuntu 16.04  
- Python 2.7.12  
- GCC 5.4.0  

**Python Library Required:**  
- Keras 1.1.1  
- Theano 0.8.2  
- h5py 2.7.1  
- numpy 1.12.0  
- scipy 0.19.1  

**Training, development, and test file:**

1. `data/*`  
    - The dataset which has been randomly generated. Used for all training, development, and test, with active learning.

**How to run:**

1. Download the dataset by running `refresh_dataset.sh` script.  
2. Training: `python train.py -tr <training_file> -o <output_folder> -t <model_type>` or simply `./run_train.sh`:  
    - `<training_file>` specifies the path to the training file (e.g., `data/foo_1m.out`)  
    - `<output_folder>` specifies the folder name for the output of the program  
    - `<model_type>` specifies the model name to be run. The best model is currently `nn` (a CNN).  
      - A few other classifiers will be available soon

