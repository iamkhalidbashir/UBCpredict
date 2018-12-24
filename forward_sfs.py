# Compare Algorithms
import pandas
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# load dataset


file_dir = os.path.dirname(__file__)

path = file_dir + "\\dataset\\ubc_train_dataset.csv"

realNames = ['AgeRecode', 'sex', 'YOD', 'martialStatus', 'grade', 'tumorSize', 'lymphNodes', 'TNinsitu', 'HT_ICDO3', 'primarySite', 'derivedAJCC',
		 'regionalNodePostive', 'class']
dataframe = pandas.read_csv(path, names=realNames)
array = dataframe.values
X = array[:,0:12]
Y = array[:,12]
# exit()
# prepare configuration for cross validation test harness
seed = 7
# prepare models
models = []
# models.append(('LR', LogisticRegression(solver='liblinear',verbose=1)))
# models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
# models.append(('SVM', SVC(gamma='auto',verbose=1)))
# evaluate each model in turn


results = []
names = []
scoring = 'accuracy'
for name, model in models:
    labels = np.arange(0,12,1);
    chosenL = []
    bestGA = 0
    print("\n" + name + ":")
    for label1 in np.arange(0,12,1):
        bestA = 0
        bestF = 0
        for label2 in labels:
            chosenL.append(label2)
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            # print(chosenL)
            cv_results = model_selection.cross_val_score(model, X[:,chosenL], Y, cv=kfold, scoring=scoring)
            a = cv_results.mean()
            if a > bestA:
                bestA = a
                bestF = label2
            chosenL.remove(label2)
        if bestGA < bestA:
            # print(bestA)
            bestGA = bestA
            chosenL.append(bestF)
            labels = np.setdiff1d(labels,[bestF])
        else:
            break
    else:
        continue
        break
    results.append(bestGA)
    names.append(name)
    msg = "%s: %f (%f) %s" % (name, cv_results.mean(), cv_results.std(), list(np.array(realNames)[chosenL]))
    print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
