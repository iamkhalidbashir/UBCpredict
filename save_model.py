# Compare Algorithms
import pandas
import os
from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

file_dir = os.path.dirname(__file__)
# load dataset
path = file_dir + "\\dataset\\ubc_train_dataset.csv"
names = ['AgeRecode', 'sex', 'YOD', 'martialStatus', 'grade', 'tumorSize', 'lymphNodes', 'TNinsitu', 'HT_ICDO3', 'primarySite', 'derivedAJCC',
		 'regionalNodePostive', 'class']

dataframe = pandas.read_csv(path, names=names)
array = dataframe.values
X = array[:,0:12]
Y = array[:,12]

# define your classifier here
# classifier = DecisionTreeClassifier() #Decision trees
# classifier = GaussianNB() #Naive Bayes
# classifier = KNeighborsClassifier() #K-nearest Neighbor
classifier = SVC(gamma='auto', kernel='rbf') #SVM

model = classifier.fit(X, Y)

print(model.predict(X[1:10,:]))
print(Y[1:10])

# filename = file_dir + '\\models\\decision_tree_model.joblib.pkl'
# filename = file_dir + '\\models\\naive_bayes_model.joblib.pkl'
# filename = file_dir + '\\models\\knn_model.joblib.pkl'
filename = file_dir + '\\models\\svm_model.joblib.pkl'

_ = joblib.dump(model, filename, compress=9)

print("\nSaved!\n")
