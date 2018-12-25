# UBCpredict

This python repository aims to develop an efficient urinary bladder cancer prediction model with machine learning techniques.

Simple steps to get it to work!

  - Clone the repository 
  - Run "py predict.py"
  - Select the model (from 1-5)
  - Select filename for test dataset
  - See the magic!

### Prediction (predict.py)
This script is used to predict a model created by (create_model.py). Some models are already created by the work done in the paper referenced at the bottom. Create a csv file have 12 features per column (see example in test_dataset.csv). The features are explained below:-
  - Age recode (value from 1-18: 1 = 1-4 years old, 2 = 5-9 years old,.., 18 = 85+ years old)
  - Sex (1 = male: 0 = female)
  - Year of diagnosis (actuall year, for years greater than 2015, just use 2015 instead)
  - Martial status (1 = not married, 0 = married,seperated,divorced)
  - Cancer grade (1-17)
  - Tumor size (0-99mm = 100, 100-199mm = 200, ..)
  - Lymph nodes (1-6)
  - Total number of insitu tumors (1,0)
  - Histologic type ICD-O-3
  - Primary site code (lookup SEER documentation)
  - Derived AJCC
  - Regional positive nodes

### Creating own model (create_model.py)

This script could be used to create your own models and contriute towards this repository.

### Classifier comparison (algo_comparision.py)

As described in the paper, 4 algorithms were compared out of which best one was used as a default model in the (predict.py) file. This script runs a comparision test and prints out the metrics for evaluation of various classifiers.

### Reference paper

The following paper was used in creation of this repository

/papers/prediction_of_the_morality_for_ubc.pdf

### License

MIT License
