'''
Created on 9 Aug 2011
Evaluating various types of classifiers for evaluating co-clustering
visual words to create visual topics. The parameters to the classifiers
will be about the best for the respective classifier. So this evaluation
will serve to compare the classifiers themselves.
@author: ag00087
'''
import numpy as np
from optparse import OptionParser
from scikits.learn.svm.classes import SVC
from scikits.learn.cross_val import StratifiedKFold
from scikits.learn.metrics.metrics import precision_recall_curve, auc, roc_curve,\
    f1_score
import sys
from scikits.learn.neighbors import NeighborsClassifier
from ash.cocluster import cocluster
parser = OptionParser()

parser.add_option('-d','--dataSet',type='string',metavar='dataSet',dest='dataSet',default='VOC2006',help='visual datasets:VOC2006,VOC2007,VOC2010,Scene15,Caltech101,Caltech256')
parser.add_option('-s','--ccScheme',type='string',metavar='ccScheme',dest='ccType',default='i',help='type of co-clustering scheme: e, i, r')
parser.add_option('-t','--nTopic',type='int',metavar='nTopic',dest='nTopic',default=1000,help='number of topics in the visual topic model')
parser.add_option('-q','--quiet',action='store_false',dest='verbose',metavar='verbose',default=True,help='verbosity')
parser.add_option('-x','--nFoldXVal',type='int',metavar='nFold',dest='nFold',default=4,help='nFold stratified cross validation for the classifier')

rootDir = '/vol/vssp/diplecs/ash/Data/'
bofDir = '/BOF/'
tempDir = '/Temp/'
cbDir = '/'
outDir = '/results/'

# global variables
catidfname = 'catidlist.txt'
cbext = '.cb'
bofext = '.bof'
evalext = '.eval'

def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.genfromtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.genfromtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap

def svmClassify(data,probability=True,kernelType='rbf',nFold=10,beta=1,nMetrics=3):
    X = data[:,:-1]
    y = data[:,-1]

    clfParamList = {'kernel': kernelType, 'gamma': 1e-3, 'C': 1, 'degree':4, 'probability':probability,'shrinking':True,'cache_size':10000}
    classifier = SVC(**clfParamList)
    
    #cross-validation
    cv = StratifiedKFold(y, k=nFold)
        
    metricstemp = np.zeros((nFold,nMetrics),np.float)
    for i, (train, test) in enumerate(cv):
        clf = classifier.fit(X[train], y[train])
        ypred = clf.predict(X[test])
        if(probability==True):
            probas_ = clf.predict_proba(X[test])
            precision, recall, thresholds = precision_recall_curve(y[test], probas_[:,1]) #@UnusedVariable
            fpr,tpr,thresholds = roc_curve(y[test],probas_[:,1]) #@UnusedVariable
            roc_auc = auc(fpr,tpr)
            pr_auc = auc(recall,precision)
        else:
            f1score = f1_score(y[test],ypred)
            fpr,tpr,thresholds = roc_curve(y[test],ypred) #@UnusedVariable
            roc_auc = auc(fpr,tpr)
            
        metricstemp[i] = [f1score,roc_auc,pr_auc]
    
    return metricstemp

def knnClassify(data,nFold=10,nNeighbor=10,nMetrics=2):
    X = data[:,:-1]
    y = data[:,-1]
    
    clfParamList = {'n_neighbors':nNeighbor,'algorithm':'auto'}
    classifier = NeighborsClassifier(**clfParamList)
    
    #cross-validation
    cv = StratifiedKFold(y, k=nFold)
        
    metricstemp = np.zeros((nFold,nMetrics),np.float)
    for i, (train, test) in enumerate(cv):
        clf = classifier.fit(X[train], y[train])
        ypred = clf.predict(X[test])
        f1score = f1_score(y[test],ypred)
        fpr,tpr,thresholds = roc_curve(y[test],ypred) #@UnusedVariable
        roc_auc = auc(fpr,tpr)
        metricstemp[i] = [f1score,roc_auc]
    
    return metricstemp
        
def ccTopic():
    (options, args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
    nRowCluster = options.nRowCluster
    nColCluster = options.nColCluster
    ccType = options.ccType
    kernelType = options.kernelType
    beta = options.beta
    figfmt = options.figfmt
    nFold = options.nFold
    nCodeword = options.nCodeword
        
    if(options.verbose): 
        print dataset,nRowCluster,nColCluster,ccType,kernelType,beta,figfmt,nFold,nCodeword
    
    dataPath = rootDir+dataset+bofDir
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    dataext = str(nCodeword)+bofext
    nCategory = len(catList)
    
    perfMean = np.zeros(nCategory)
    perfStd = np.zeros(nCategory)
    
    for iCategory,catName in enumerate(catList):
        fileName = dataPath+catName+dataext
        catpos = np.genfromtxt(fileName,dtype=np.int)
        if(options.verbose): print catName
        catpos = catpos[:,:nCodeword+1]
        catpos[:,nCodeword] = 1
        #read the category data of remaining classes
        
        for cats in catList:
            if(cats!=catName):
                firstvisit = True
                if(firstvisit):
                    catneg = np.genfromtxt(fileName,dtype=np.int)
                    firstvisit = False
                else : 
                    catneg = np.concatenate((catneg,np.genfromtxt(fileName,dtype=np.int)),axis=0)
        #sample the negative data to have equal size as the positive
        nPos = catpos.shape[0]
        nNeg = catneg.shape[0]
        catneg = catneg[np.random.randint(0,nNeg,nPos),:] #catneg
        catneg = catneg[:,:nCodeword+1]
        catneg[:,nCodeword] = 0
        #combine positive and negative data
        bofData = np.concatenate((catpos,catneg),axis=0)
        if(options.verbose): print 'co-clustering...'
        ccData = cocluster.coclust(bofData, dataset, nRowCluster, nColCluster, ccType)
                
        ccCol = np.array([int(i) for i in ccData[1].split()])
        tempCC = np.zeros((bofData.shape[0],nColCluster))
        for i in np.arange(bofData.shape[0]):
            for j in sorted(set(ccCol)):
                tempCC[i,j] = np.sum(bofData[i,ccCol==j])
     
        botData = np.vstack((tempCC.T,bofData[:,-1])).T
        
        if(options.verbose): print 'classifying...'
#        catPerf = classify.ccClassify(botData, kernelType, nFold, beta, nMetrics)
        catPerf = knnClassify(botData, 10, nFold, beta)
        perfMean[iCategory] = np.mean(catPerf)
        perfStd[iCategory] = np.std(catPerf) 

    if(options.verbose):
        print perfMean
        print perfStd
    #plotresult.ccPlot(dataset, catList, perfMean, perfStd, figfmt, 'BoT')
    return [perfMean,perfStd]

if __name__ == '__main__':
    pass