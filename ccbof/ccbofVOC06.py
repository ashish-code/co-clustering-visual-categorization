'''
Created on 17 Jul 2011

@author: ag00087
'''

import sys
import os
import numpy as np
from scikits.learn import svm
from scikits.learn.metrics import roc_curve,precision_recall_curve,auc
from scikits.learn.cross_val import StratifiedKFold
import matplotlib.pyplot as plt


#globals
# data paths
dataSet_default = 'VOC2006'
rootDir = '/vol/vssp/diplecs/ash/Data/'
bofDir = '/BOF/'
tempDir = '/Temp/'
cbDir = '/'
outDir = '/results/'

cctype_default = 'i'
nClassifyMetrics_default = 1
nFoldCrossValidation_default = 2
classifierKernelType_default = 'linear'
nTopic_default = 1000
nFold = 4
nCodeword = 10000
# global variables
dataSets = ['Caltech101','Caltech256','Scene15','VOC2006','VOC2007','VOC2010']
catidfname = 'catidlist.txt'
cbext = '.cb'
bofext = '.bof'
evalext = '.eval'
metrics = ['auc-roc','map']
nMetrics = len(metrics)


def getargs():
    try:
        dataset = sys.argv[1]
        if(dataset in dataSets == False):
            print 'dataset: %s not recognised.' % (dataset)
    except:
        dataset = dataSet_default
        print 'dataset could not be read, using %s ' % (dataset)
               
    try:
        nTopic = int(sys.argv[2])
        print 'nTopic: %d ' % (nTopic)
    except:
        nTopic = nTopic_default
        print 'using default value... nTopic: %d ' % (nTopic)
    
    try:
        cctype = sys.argv[3]
        if(cctype in ['e','i','r'] == False): cctype = cctype_default
        print 'cctype is %s ' % (cctype)
    except:
        cctype = cctype_default
        print 'cctype is %s ' % (cctype)   
    
    try:
        classifierKernelType = sys.argv[4]
        print 'classifier kernel type: %s' % (classifierKernelType)
    except:
        classifierKernelType = classifierKernelType_default
        print 'using default classifier kernel type: %s' % (classifierKernelType)
        
    return dataset,nTopic,cctype,classifierKernelType

def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.genfromtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.genfromtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap    

def coclust(bofData,nTopic,ccType,dataset):
    tempPath = rootDir+dataset+tempDir
    tempBOFDataPath = tempPath+'tempBOFData'
    tempBOFDataDimPath = tempPath+'tempBOFData_dim'
    tempBOTPath = tempPath+'tempBOT.txt'
    np.savetxt(tempBOFDataPath,bofData[:,:nCodeword],fmt='%d',delimiter=' ')
    np.savetxt(tempBOFDataDimPath,bofData[:,:nCodeword].shape,fmt='%d',delimiter=' ')
    cmdPath = '/vol/vssp/diplecs/ash/code/cocluster/'
    cmd = 'cocluster-linux'
    args = ' -A %s -R %d -C %d -I d s %s -O c s 0 o %s' % (ccType,1,nTopic,tempBOFDataPath,tempBOTPath)
    cwd = os.getcwd()
    os.chdir(cmdPath)
    os.system(cmd + args)
    os.chdir(cwd)
    botFile = open(tempBOTPath,'r')
    ccData = botFile.readlines()
    botFile.close()
    #botR = np.array([int(i) for i in botData[0].split()])
    botC = np.array([int(i) for i in ccData[1].split()])
    #print botC
    tempBOT = np.zeros((bofData.shape[0],nTopic))
    for i in np.arange(bofData.shape[0]):
        for j in sorted(set(botC)):
            tempBOT[i,j] = np.sum(bofData[i,botC==j])
 
    botData = np.vstack((tempBOT.T,bofData[:,nCodeword])).T
    return botData

def catClassify(botData,kernelType,nTopic):
    X = botData[:,:nTopic]
    y = botData[:,nTopic]
    
           
    # Run classifier
    #    classifier = svm.SVC(kernel='linear', probability=True)
    
    classifier = svm.NuSVC(probability=True)

    #cross-validation
    cv = StratifiedKFold(y, k=nFold)
    #select classifier
    #classifier = svm.SVC(kernel=kernelType, probability=True)
        
    metricstemp = np.zeros((nFold,nMetrics),np.float)
    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        #fpr, tpr, thresholds = roc_curve(y[test], probas_[:,1]) #@UnusedVariable
        #roc_auc = auc(fpr, tpr)
        precision, recall, thresholds = precision_recall_curve(y[test], probas_[:,1]) #@UnusedVariable
        pr_auc = auc(recall,precision)
        metricstemp[i] = [pr_auc]
    
    return [np.mean(metricstemp),np.std(metricstemp) ]

def plotPerf(dataset,kernelType,perfMean,perfStd,catmap):
    nCategories = len(catmap.keys())
    outPath = rootDir + dataset + outDir + '%s%s%s%s'%('perfBagofTopic',dataset,kernelType,'.png') 
    plt.figure()
    ax = plt.subplot(111)
    plt.errorbar(np.arange(1,(nCategories+1)), perfMean, perfStd, fmt = '-', ecolor='k', elinewidth=1, marker = 'x', label = 'MeanAveragePrecision')

    #plt.errorbar(np.arange(1,(nCategories+1)), perfMean[:,1], perfStd[:,1], fmt = '--', ecolor='k', elinewidth=1, marker = 'x', label = 'MAP')
    plt.xlabel('Visual Categories')
    plt.ylabel('Performance Metric')
    plt.title('Bag-of-Topics Performance: %s : %s' % (dataset,kernelType))
    plt.legend(loc="lower right")
    #plt.ylim([0.0,1.0])
    ax.set_xticks(np.arange(1,(nCategories+2)))
    ax.set_xticklabels(catmap.keys(),rotation=30,size='small',ha='center')
    #plt.setp(catmap.keys(), rotation=30, fontsize=10)
    plt.savefig(outPath,format='png')
    plt.show()
    plt.close()

def main():
    dataset,nTopic,ccType,classifierKernelType = getargs()
    dataPath = rootDir+dataset+bofDir
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    dataext = bofext
    nCategory = len(catmap.keys())
    perfMean = np.zeros(nCategory)
    perfStd = np.zeros(nCategory)
    for iCategory,catName in enumerate(catList):
        fileName = dataPath+catName+'10000'+dataext
        catpos = np.genfromtxt(fileName,dtype=np.int)
        print catpos.shape
        catpos = catpos[:,:nCodeword+1]
        catpos[:,nCodeword] = 1
        #read the category data of remaining classes
        
        
        for cats in catmap.keys():
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
        print 'coclustering...'
        botData = coclust(bofData,nTopic,ccType,dataset)
        print 'classifying...'
        catPerf = catClassify(botData,classifierKernelType,nTopic)
        perfMean[iCategory] = catPerf[0]
        perfStd[iCategory] = catPerf[1] 

    print 'plotting performance results...'
    print perfMean
    print perfStd
    plotPerf(dataset,classifierKernelType,perfMean,perfStd,catmap)
    
if __name__ == '__main__':
    main()