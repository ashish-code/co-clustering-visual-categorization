'''
Created on 25 Jul 2011

@author: ag00087
'''
import numpy as np
from sklearn import svm
from sklearn.metrics import precision_recall_curve,auc
from sklearn.cross_val import StratifiedKFold
from sklearn.neighbors import NeighborsClassifier
from sklearn.metrics.metrics import fbeta_score

def ccClassify(data,kernelType='rbf',nFold=10,beta=1,nMetrics=1):
    X = data[:,:-1]
    y = data[:,-1]
               
    # Run classifier
    #classifier = svm.SVC(kernel='linear', probability=True)
    classifier = svm.NuSVC(probability=True)
    #cross-validation
    cv = StratifiedKFold(y, k=nFold)
        
    metricstemp = np.zeros((nFold,nMetrics),np.float)
    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        precision, recall, thresholds = precision_recall_curve(y[test], probas_[:,1]) #@UnusedVariable
        pr_auc = auc(recall,precision)
        metricstemp[i] = [pr_auc]
    
    return metricstemp

def svmClassify(data,kernelType='rbf',nFold=10,beta=1.0,nMetrics=1):
    X = data[:,:-1]
    y = data[:,-1]
               
    # Run classifier
    #classifier = svm.SVC(kernel='linear', probability=True)
    classifier = svm.NuSVC(kernel=kernelType)
    #cross-validation
    cv = StratifiedKFold(y, k=nFold)
        
    avgprec = np.zeros(nFold)
    for icv,(train, test) in enumerate(cv):
        clf = classifier.fit(X[train], y[train])
        ypred = clf.predict(X[test])
        avgprec[icv] = fbeta_score(y[test],ypred,beta)
    return avgprec

def knnClassify(data,n_neighbors=10,nFold=10,beta=1.0,nMetrics=1):
    X = data[:,:-1]
    y = data[:,-1]
    clfParamList = {'n_neighbors':10,'algorithm':'auto'}
    classifier = NeighborsClassifier(**clfParamList)
    cv = StratifiedKFold(y, k=nFold)
    avgprec = np.zeros(nFold)
    for icv,(train, test) in enumerate(cv):
        clf = classifier.fit(X[train], y[train])
        ypred = clf.predict(X[test])
        avgprec[icv] = fbeta_score(y[test],ypred,beta)
    return avgprec

if __name__ == '__main__':
    pass