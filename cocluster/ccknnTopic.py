'''
Created on 7 Aug 2011
Run co-clustering of visual words to discover visual topics and also
run baseline BoF implementation. Compare results to show topics are
superior to visual words for category classification.
@author: ag00087
'''

import numpy as np
import sys
import cocluster,classify
from ccnblplt import plotresult
from optparse import OptionParser
#from scikits.learn.svm.classes import SVC
from sklearn.cross_val import StratifiedKFold
from sklearn.metrics.metrics import f1_score
from sklearn.neighbors import NeighborsClassifier

parser = OptionParser()
parser.add_option('-d','--dataset',type='string',metavar='dataset',dest='dataset',default='VOC2006',help='the dataset')
parser.add_option('-r','--rowClusters',type='int',metavar='nRowCluster',dest='nRowCluster',default=10,help='number of row clusters')
parser.add_option('-c','--colClusters',type='int',metavar='nColCluster',dest='nColCluster',default=1000,help='number of column clusters')
parser.add_option('-t','--ccType',type='string',metavar='ccType',dest='ccType',default='e',help='type of co-clustering scheme: e,i,r')
parser.add_option('-k','--kernel',type='string',metavar='kernelType',dest='kernelType',default='rbf',help='type of svm kernel, linear,rbf,poly,sigmoid')
parser.add_option('-x','--nFoldXVal',type='int',metavar='nFold',dest='nFold',default=4,help='number of stratified cross validation iterations')
parser.add_option('-p','--figformat',type='string',metavar='figfmt',dest='figfmt',default='png',help='type of output graph image, png, svg, jpg')
parser.add_option('-b','--beta',type='float',metavar='beta',dest='beta',default=1.0,help='f-beta value, default is 1.0')
parser.add_option('-w','--nWord',type='int',metavar='nWord',dest='nCodeword',default=10000,help='codewords')
parser.add_option("-q", "--quiet", action="store_false", dest="verbose", default=True)

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
nMetrics = 1

def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.genfromtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.genfromtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap    


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
        catPerf = classify.knnClassify(botData, 10, nFold, beta, nMetrics)
        perfMean[iCategory] = np.mean(catPerf)
        perfStd[iCategory] = np.std(catPerf) 

    if(options.verbose):
        print perfMean
        print perfStd
    #plotresult.ccPlot(dataset, catList, perfMean, perfStd, figfmt, 'BoT')
    return [perfMean,perfStd]

def blTopic():
    (options, args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
    
#    kernelType = options.kernelType
    nFold = options.nFold
    nCodeword = options.nColCluster   
    
    dataPath = rootDir+dataset+bofDir
    catmap = getCatMap(dataset)
    catList = catmap.keys()
#    dataext = str(nCodeword)+bofext
    dataext = bofext
    nCategory = len(catList)
    
    perfMean = np.zeros(nCategory)
    perfStd = np.zeros(nCategory)
    
    for iCat,catname in enumerate(catList):
        print catname
        #read the category data which will positive
        fname = dataPath+catname+dataext
        catpos = np.genfromtxt(fname,dtype=np.int) # catpos
        catpos = catpos[:,:nCodeword+1]
        catpos[:,nCodeword] = 1
        #read the category data of remaining classes
        for cats in catList:
            if(cats!=catname):
                firstvisit = True
                if(firstvisit):
                    catneg = np.genfromtxt(fname,dtype=np.int)
                    firstvisit = False
                else : 
                    catneg = np.concatenate((catneg,np.genfromtxt(fname,dtype=np.int)),axis=0)
        #sample the negative data to have equal size as the positive
        nPos = catpos.shape[0]
        nNeg = catneg.shape[0]
        catneg = catneg[np.random.randint(0,nNeg,nPos),:] #catneg
        catneg = catneg[:,:nCodeword+1]
        catneg[:,nCodeword] = 0
        #combine positive and negative data
        data = np.concatenate((catpos,catneg),axis=0)
        #shuffle the rows to aid in random selection of train and test
        np.random.shuffle(data)
        
        X = data[:,:nCodeword]
        y = data[:,nCodeword]
        
#        clfParamList = {'kernel': kernelType, 'gamma': 1e-3, 'C': 1, 'degree':4, 'probability':True,'shrinking':True,'cache_size':1000}
#        classifier = SVC(**clfParamList)
        clfParamList = {'n_neighbors':10,'algorithm':'auto'}
        classifier = NeighborsClassifier(**clfParamList)
        cv = StratifiedKFold(y, k=nFold)
        avgprec = np.zeros(nFold)
        for icv,(train, test) in enumerate(cv):
            clf = classifier.fit(X[train], y[train])
#            probas_ = clf.predict_proba(X[test])
#            precision, recall, thresholds = precision_recall_curve(y[test], probas_[:,1]) #@UnusedVariable
#            avgprec[icv] = auc(recall,precision)
            ypred = clf.predict(X[test])
            avgprec[icv] = f1_score(y[test],ypred)
        perfMean[iCat] = np.mean(avgprec)
        perfStd[iCat] = np.std(avgprec)
    return [perfMean,perfStd]
    
def ccnblTopic():
    (options, args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
    ccType = options.ccType
    figfmt = options.figfmt
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    ccResult = ccTopic()
    blResult = blTopic()
    plotresult(dataset,catList,ccResult,blResult,figfmt,'CoClusterBOFKNN',ccType)
    
if __name__ == '__main__':
    ccnblTopic()
    