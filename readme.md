ProgramVerificationClassifier
===================================
Given a set of input to a program or function, the model should learn and output either valid (1) or invalid (0).
Invalid inputs will result in a program crash.

The repository is publicly available at https://github.com/yulonglong/ProgramverificationClassifier  

**Requirements and Environment:**  
- Ubuntu 16.04  
- Python 2.7.12  
- GCC 5.4.0  

**Python Library Used:**  
- Keras 1.1.1  
- Theano 0.8.2  
- h5py 2.6.0  
- numpy 1.12.0  
- scipy 0.19.0  

**Training, development, and test file:**

1. `data/*`  
    - The dataset which has been randomly generated. Used for all training, development, and test, with active learning.

**How to run:**

1. Training: `python main.py -tr <training_file> -o <output_folder> -t <model_type>` or simply `./train_nn.sh`:  
    - `<training_file>` specifies the path to the training file, in this case it is `data/train.csv`  
    - `<output_folder>` specifies the folder name for the output of the program  
    - `<model_type>` specifies the model name to be run. The best model is currently `nn` (a CNN).  
      - A few other classifiers will be `LogisticRegression` and `SVC` (SVM). Refer to the code for the list of possible model types.  
2. Testing: `python main.py -tr <training_file> -o <output_folder> -t <model_type> --test` or simply `./test_nn.sh`:  
    - Same explanation as above, training on the fly and predicting after finish training and validating  
