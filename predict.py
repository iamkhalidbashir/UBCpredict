# Compare Algorithms
import pandas
import os
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier

file_dir = os.path.dirname(__file__)
# load dataset
model_name = 'svm_model_linear'
input_filename = "test_dataset.csv"

def init():
	global model_name,input_filename
	print("\n\n---> Welcome to UBCpredict <---\n\n\
	Please choose a model:\n\n\
	1) Naive Bayes\n\
	2) K-Nearest Neighbor\n\
	3) Decision Trees\n\
	4) Support vector machine (SVM) (linear)\n\
	5) Support vector machine (SVM) (RBF)\n\n\
	Option: ")
	option = int(input())

	if option == 1:
		model_name = "naive_bayes_model"
	elif option == 2:
		model_name = "knn_model"
	elif option == 3:
		model_name = "decision_tree_model"
	elif option == 4:
		model_name = "svm_model_linear"
	elif option == 5:
		model_name = "svm_model_rbf"

	print("\n\nFilename: test_dataset.csv?\n:")
	filename = input()
	if filename != "":
		input_filename = filename

def get_predictdata():
	global model_name,input_filename
	names = ['AgeRecode', 'sex', 'YOD', 'martialStatus', 'grade', 'tumorSize', 'lymphNodes', 'TNinsitu', 'HT_ICDO3', 'primarySite', 'derivedAJCC',
			 'regionalNodePostive', 'class']

	path = file_dir + "\\" + input_filename
	dataframe = pandas.read_csv(path, names=names)
	array = dataframe.values
	X = array[:,0:12]
	return X

def predict(X):
	model_path = file_dir + '\\models\\' + model_name + ".joblib.pkl"
	model = joblib.load(model_path)
	prediction = model.predict(X)
	print("\n\nPrediction result:\n\n")
	i = 1
	for p in prediction:
		if p == 0:
			print(str(i) + ") The patient will meet his creator soon!")
		elif p == 1:
			print(str(i) + ") The patient will survive for atleast 2.5 years!")
		elif p == 2:
			print(str(i) + ") The patient will survive for atleast 5 years!")
		elif p == 3:
			print(str(i) + ") The patient will survive for atleast 7.5 years!")
		elif p == 4:
			print(str(i) + ") The patient will survive for atleast 10 years!")
		i = i + 1

init()
X = get_predictdata()
predict(X)
