'''
Created on 14 Jul 2011

@author: ag00087
'''
import sys
import os
import numpy as np
from scipy.cluster.vq import vq
from scikits.learn import svm
from scikits.learn.metrics import roc_curve,precision_recall_curve,auc
from scikits.learn.cross_val import StratifiedKFold
import matplotlib.pyplot as plt

#globals
# data paths
dataSet_default = 'VOC2006'
rootDir = '/vol/vssp/diplecs/ash/Data/'
featureDataDir = '/FeatureMatrix/'
bofDir = '/BOF/'
tempDir = '/Temp/'
cbDir = '/'
outDir = '/results/'
# arguments default values
nClusterSample_default = 500000
nCodeword_default = 1000
cctype_default = 'e'
nClassifyMetrics_default = 2
nFoldCrossValidation_default = 10
classifierKernelType_default = 'linear'
nTopic_default = 13
nFold = 4
# global variables
dataSets = ['Caltech101','Caltech256','Scene15','VOC2006','VOC2007','VOC2010']
catidfname = 'catidlist.txt'
cbext = '.cb'
bofext = '.bof'
evalext = '.eval'
metrics = ['auc-roc','map']
nMetrics = len(metrics)

def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.genfromtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.genfromtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap

def coclust(clusterData,dataset,nR,nC,cctype):
    tempPath = rootDir+dataset+tempDir
    tempClusterDataPath = tempPath+'tempClusterData'
    tempClusterDataDimPath = tempPath+'tempClusterData_dim'
    tempCCPath = tempPath+'tempCC.txt'
    np.savetxt(tempClusterDataPath,clusterData,fmt='%d',delimiter=' ')
    np.savetxt(tempClusterDataDimPath,clusterData.shape,fmt='%d',delimiter=' ')
    cmdPath = '/vol/vssp/diplecs/ash/code/cocluster/'
    cmd = 'cocluster-linux'
    args = ' -A %s -R %d -C %d -I d s %s -O c s 0 o %s' % (cctype,nR,nC,tempClusterDataPath,tempCCPath)
    cwd = os.getcwd()
    os.chdir(cmdPath)
    os.system(cmd + args)
    os.chdir(cwd)
    ccFile = open(tempCCPath,'r')
    ccData = ccFile.readlines()
    ccFile.close()
    ccR = np.array([int(i) for i in ccData[0].split()])
    ccC = np.array([int(i) for i in ccData[1].split()])
    cctemp1 = np.zeros((clusterData.shape[0],nC))
    centroids = np.zeros((nR,nC))
    for i in np.arange(clusterData.shape[0]):
        for j in sorted(set(ccC)):
            cctemp1[i,j] = np.linalg.norm(clusterData[i,ccC==j], 2)
    for i in sorted(set(ccR)):
        centroids[i,:] = np.mean(cctemp1[ccR==i,:],0)
    
    return centroids,ccC
    
def writebof(catName,dataset,codeBook,colCluster,nCodeword,dataext):
    dataPath = rootDir+dataset+featureDataDir
    catFilePath = dataPath+catName+dataext
    catData = np.genfromtxt(catFilePath,dtype=np.int8,usecols=np.arange(2,130))
    catImgId = np.genfromtxt(catFilePath,dtype=np.int,usecols=np.arange(130,131))
    catId = np.genfromtxt(catFilePath,dtype=np.int,usecols=np.arange(131,132))[0]
    ImgId = np.unique(catImgId)
    #collate the columns
    catD = np.zeros((catData.shape[0],len(set(colCluster))))
    for i in np.arange(catData.shape[0]):
        for j in sorted(set(colCluster)):
            catD[i,:] = np.linalg.norm(catData[i,colCluster==j], 2)
    
    # vq the data
    [catLabel,catDist] = vq(catD,codeBook) #@UnusedVariable
    # write the bag-of-features to file
    catboffilepath = rootDir+dataset+bofDir+catName+'_cc'+bofext
    catboffile = open(catboffilepath,'w')
    imgcount=0
    for imgid in ImgId:
        imgLabel = catLabel[catImgId==imgid]
        [hist,edges] = np.histogram(imgLabel,nCodeword) #@UnusedVariable
        if imgcount==0:
            dataout = np.hstack((hist.T,imgid,catId))
        else:
            dataout = np.vstack((dataout,np.hstack((hist.T,imgid,catId))))
        imgcount+=1
    np.savetxt(catboffile, dataout, fmt='%d', delimiter=' ', )
    catboffile.close()
    
def catClassify(dataPath,catname,kernelType,dataext,catmap,nTopic):
    #read the categoy data which will positive
    fname = dataPath+catname+dataext
    catpos = np.genfromtxt(fname,dtype=np.int) # catpos
    catpos = catpos[:,:nTopic+1]
    catpos[:,nTopic] = 1
    #read the category data of remaining classes
    for cats in catmap.keys():
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
    catneg = catneg[:,:nTopic+1]
    catneg[:,nTopic] = 0
    #combine positive and negative data
    data = np.concatenate((catpos,catneg),axis=0)
    #shuffle the rows to aid in random selection of train and test
    np.random.shuffle(data)
    
    X = data[:,:nTopic]
    y = data[:,nTopic]
    
    #cross-validation
    cv = StratifiedKFold(y, k=nFold)
    #select classifier
    classifier = svm.SVC(kernel=kernelType, probability=True)
    metricstemp = np.zeros((nFold,nMetrics),np.float)
    for i, (train, test) in enumerate(cv):
        probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:,1]) #@UnusedVariable
        roc_auc = auc(fpr, tpr)
        precision, recall, thresholds = precision_recall_curve(y[test], probas_[:,1]) #@UnusedVariable
        pr_auc = auc(recall,precision)
        metricstemp[i] = [roc_auc,pr_auc]
    
    return [np.mean(metricstemp,axis=0),np.std(metricstemp,axis=0) ]
      
def plotPerf(dataset,kernelType,perfMean,perfStd,catmap):
    nCategories = len(catmap.keys())
    outPath = rootDir + dataset + outDir + '%s%s%s%s'%('perfCoCluster',dataset,kernelType,'.png') 
    plt.figure()
    ax = plt.subplot(111)
    plt.errorbar(np.arange(1,(nCategories+1)), perfMean[:,0], perfStd[:,0], fmt = '-', ecolor='k', elinewidth=1, marker = 'x', label = 'AUC-ROC')
    plt.errorbar(np.arange(1,(nCategories+1)), perfMean[:,1], perfStd[:,1], fmt = '--', ecolor='k', elinewidth=1, marker = 'x', label = 'MAP')
    plt.xlabel('Visual Categories')
    plt.ylabel('Performance Metric')
    plt.title('BOF Performance: %s : %s' % (dataset,kernelType))
    plt.legend(loc="lower right")
    plt.ylim([0.0,1.0])
    ax.set_xticks(np.arange(1,(nCategories+2)))
    ax.set_xticklabels(catmap.keys(),rotation=30,size='small',ha='center')
    #plt.setp(catmap.keys(), rotation=30, fontsize=10)
    plt.savefig(outPath,format='png')
    plt.show()
    plt.close()
    
def getargs():
    try:
        dataset = sys.argv[1]
        if(dataset in dataSets == False):
            print 'dataset: %s not recognised.' % (dataset)
    except:
        dataset = dataSet_default
        print 'dataset could not be read, using %s ' % (dataset)
        
    try:
        nCodeword = int(sys.argv[2])
        print 'nCodeword: %d ' % (nCodeword)
    except:
        nCodeword = nCodeword_default
        print 'using default value... nCodeword: %d ' % (nCodeword)
        
    try:
        nTopic = int(sys.argv[3])
        print 'nTopic: %d ' % (nTopic)
    except:
        nTopic = nTopic_default
        print 'using default value... nTopic: %d ' % (nTopic)
    
    try:
        cctype = sys.argv[4]
        if(cctype in ['e','i','r'] == False): cctype = cctype_default
        print 'cctype is %s ' % (cctype)
    except:
        cctype = cctype_default
        print 'cctype is %s ' % (cctype)
        
    try:
        nClusterSample = int(sys.argv[5])
        print 'nClusterSamples: %d' % (nClusterSample)
    except:
        nClusterSample = nClusterSample_default
        print 'using default value... nClusterSamples: %d' % (nClusterSample)
    
    try:
        classifierKernelType = sys.argv[6]
        print 'classifier kernel type: %s' % (classifierKernelType)
    except:
        classifierKernelType = classifierKernelType_default
        print 'using default classifier kernel type: %s' % (classifierKernelType)
        
    return dataset,nCodeword,nTopic,cctype,nClusterSample,classifierKernelType
    
def main():
    
    dataset,nCodeword,nTopic,cctype,nClusterSample,classifierKernelType = getargs()
    
    dataPath = rootDir+dataset+featureDataDir
    catmap = getCatMap(dataset)
    catFileList = os.listdir(dataPath)
    dataext = '.'+catFileList[0].split('.')[1]
    catList = [catfilename.split('.')[0] for catfilename in catFileList]
    nCategory = len(catList)
    nSamplePerCategory = int(np.round(nClusterSample/nCategory))
    # sample data from all categories
    print 'sampling data...'
    clusterData = None
    for catName in (catList):
        catFilePath = dataPath+catName+dataext
        catData = np.genfromtxt(catFilePath,dtype=np.int8,usecols=np.arange(2,130))
        if(catData.shape[0] <= nSamplePerCategory):
            catSample = catData
        else:
            rndsample = np.random.randint(0,catData.shape[0],nSamplePerCategory)
            catSample = catData[rndsample,:]
        if(clusterData == None):
            clusterData = catSample
        else:
            clusterData = np.concatenate((clusterData,catSample),axis=0)
    
    # debugging constructs
    print clusterData.shape
    # co-cluster the data
    print 'computing cocluster...'
    codeBook,colCluster = coclust(clusterData,dataset,nCodeword,nTopic,cctype)
      
    print 'CodeBook: '
    print codeBook
    print 'colcluster: '
    print colCluster
    # compute bag-of-features for all categories
    print 'writing bof...'
    for catName in catList:
        writebof(catName,dataset,codeBook,colCluster,nCodeword,dataext)
    
    perfMean = np.arange(len(catmap.keys()))
    perfStd = np.arange(len(catmap.keys()))
    
    print 'computing classification perf...'
    for iCat,catname in enumerate(catmap.keys()):
        catPerf = catClassify(dataPath,catname,classifierKernelType,dataext,catmap,nTopic)
        perfMean[iCat] = catPerf[0]
        perfStd[iCat] = catPerf[1] 
    
    print 'plotting performance results...'
    plotPerf(dataset,classifierKernelType,perfMean,perfStd,catmap)
    
if __name__ == '__main__':
    main()