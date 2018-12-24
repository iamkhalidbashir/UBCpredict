# Compare Algorithms
import pandas
import os
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix, make_scorer
# load dataset

file_dir = os.path.dirname(__file__)

path = file_dir + "\\dataset\\ubc_train_dataset.csv"
names = ['AgeRecode', 'sex', 'YOD', 'martialStatus', 'grade', 'tumorSize', 'lymphNodes', 'TNinsitu', 'HT_ICDO3', 'primarySite', 'derivedAJCC',
		 'regionalNodePostive', 'class']
# names = ['sex', 'TNinsitu', 'class']
dataframe = pandas.read_csv(path, names=names)
array = dataframe.values
X = array[:,0:12]
Y = array[:,12]
# exit()
# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
models.append(('NB', GaussianNB()))
# models.append(('LR', LogisticRegression(solver='liblinear',verbose=1)))
# models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
# scoring = {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score)}
# scoring = "confusion_matrix"
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
def prec(y_true, y_pred): return precision_score(y_true, y_pred, average='weighted')
def acur(y_true, y_pred): return accuracy_score(y_true, y_pred)
def reca(y_true, y_pred): return recall_score(y_true, y_pred, average='weighted')
def f1(y_true, y_pred): return f1_score(y_true, y_pred, average='weighted')

scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn),
		   "prec": make_scorer(prec), "acur": make_scorer(acur),
		   "reca": make_scorer(reca), "f1": make_scorer(reca)}

print("Started... \n")

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_validate(model, X, Y, cv=kfold, scoring=scoring, verbose=1)
	results.append(cv_results['test_acur'])
	names.append(name)
	# print(cv_results)

	accuracy = cv_results['test_acur'].mean()
	accuracy_std = cv_results['test_acur'].std()

	recall = cv_results['test_reca'].mean()
	recall_std = cv_results['test_reca'].std()

	f1 = cv_results['test_f1'].mean()
	f1_std = cv_results['test_f1'].std()

	precision = cv_results['test_prec'].mean()
	precision_std = cv_results['test_prec'].std()

	f1_2 = 2*((precision*recall)/(recall+precision))
	f1_2_std = 2*((precision_std*recall_std)/(recall_std+precision_std))

	msg = "%s: accuracy: %f (%f) | precision: %f (%f) | recal: %f (%f) | f1: %f (%f) | f1(2): %f (%f)" % (name, accuracy, accuracy_std,
	precision, precision_std, recall, recall_std, f1, f1_std, f1_2, f1_2_std)
	print(msg)
# boxplot algorithm comparison
# exit()
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
