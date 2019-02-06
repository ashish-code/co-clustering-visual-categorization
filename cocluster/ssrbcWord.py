'''
Created on 17 Apr 2012

@author: ag00087
'''

import numpy as np
import sys
import cocluster,classify
#from ccnblplt import plotresult
from optparse import OptionParser
from scipy.cluster.vq import vq
#from scikits.learn.svm.classes import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics.metrics import f1_score
from sklearn.neighbors import NeighborsClassifier
import shelve
from ccnblplt import plotWordResult

rootDir = '/vol/vssp/diplecs/ash/Data/'
dataDir = '/FeatureMatrix/'
bofDir = '/BOF/'
tempDir = '/Temp/'
cbDir = '/'
#outDir = '/results/'
outDir = '/results/dagm/'
# global variables
catidfname = 'catidlist.txt'
cbext = '.cb'
bofext = '.bof'
evalext = '.eval'
nMetrics = 1
descdim = {'sift':128}

parser = OptionParser()
parser.add_option('-d','--dataset',type='string',metavar='dataset',dest='dataset',default='VOC2006',help='the dataset')
parser.add_option('-t','--ccType',type='string',metavar='ccType',dest='ccType',default='r',help='type of co-clustering scheme: e,i,r')
parser.add_option('-k','--kernel',type='string',metavar='kernelType',dest='kernelType',default='linear',help='type of svm kernel, linear,rbf,poly,sigmoid')
parser.add_option('-x','--nFoldXVal',type='int',metavar='nFold',dest='nFold',default=10,help='number of stratified cross validation iterations')
parser.add_option('-p','--figformat',type='string',metavar='figfmt',dest='figfmt',default='pdf',help='type of output graph image, png, svg, jpg')
parser.add_option('-b','--beta',type='float',metavar='beta',dest='beta',default=1.0,help='f-beta value, default is 1.0')
parser.add_option("-q", "--quiet", action="store_false", dest="verbose",default=True,help='verbosity')
parser.add_option('-f','--featureDescriptor',type='string',metavar='desc',dest='desc',default='sift',help='feature descriptor')
parser.add_option('-s','--nclusterSample',type='int',metavar='clusterSampleSize',dest='nClusterSample',default=20000,help='number of samples for clustering')
parser.add_option('-w','--nCodeword',type='int',metavar='nCodeword',dest='nCodeword',default=1000,help='codewords')
parser.add_option('-l','--lowerDim',type='int',metavar='lowerDim',dest='lowerDim',default=10,help='lower dimension of descriptor')

def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.loadtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.loadtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap    

def collateClusterData(dataPath,dataext,catList,nSamplePerCategory,dim):
    clusterData = None
    for catName in (catList):
        catFilePath = dataPath+catName+dataext
        catData = np.loadtxt(catFilePath,dtype=np.int16,usecols=np.arange(2,dim+2))
        if(catData.shape[0] <= nSamplePerCategory):
            catSample = catData
        else:
            rndsample = np.random.randint(0,catData.shape[0],nSamplePerCategory)
            catSample = catData[rndsample,:]
        if(clusterData == None):
            clusterData = catSample
        else:
            clusterData = np.concatenate((clusterData,catSample),axis=0)
    return clusterData

def writebof(dataset,catList,codebook,colCluster,nVisualWord,desc,ccType,nCodeword,lDim):
    for catName in catList:
        dataext = '.'+desc
        dim = descdim.get(desc)
        catFilePath = rootDir+dataset+dataDir+catName+dataext
        catData = np.genfromtxt(catFilePath,dtype=np.int8,usecols=np.arange(2,dim+2))
        catImgId = np.genfromtxt(catFilePath,dtype=np.int,usecols=np.arange(dim+2,dim+3))
        catId = np.genfromtxt(catFilePath,dtype=np.int,usecols=np.arange(dim+3,dim+4))[0]
        ImgId = np.unique(catImgId)
        #collate the columns
        nCol = codebook.shape[1]
        catD = np.zeros((catData.shape[0],nCol))
        for i in np.arange(catData.shape[0]):
            for j in sorted(set(colCluster)):
                catD[i,j] = np.linalg.norm(catData[i,colCluster==j], 2)
        
        # vq the data
        print catD.shape
        print codebook.shape
        [catLabel,catDist] = vq(catD,codebook) #@UnusedVariable
        # write the bag-of-features to file
        catboffilepath = rootDir+dataset+bofDir+catName+str(ccType)+str(nCodeword)+str(lDim)+bofext
        catboffile = open(catboffilepath,'w')
        imgcount=0
        for imgid in ImgId:
            imgLabel = catLabel[catImgId==imgid]
            [hist,edges] = np.histogram(imgLabel,nVisualWord) #@UnusedVariable
            if imgcount==0:
                dataout = np.hstack((hist.T,imgid,catId))
            else:
                dataout = np.vstack((dataout,np.hstack((hist.T,imgid,catId))))
            imgcount+=1
        np.savetxt(catboffile, dataout, fmt='%d', delimiter=' ', )
        catboffile.close()
'''
co-cluster the visual feature descriptor. This simulates simultaneous clustering
and dimensionality reduction.
'''
def ccWord():
    (options,args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
    nRowCluster = options.nCodeword
    nColCluster = options.lowerDim
    ccType = options.ccType
    kernelType = options.kernelType
    beta = options.beta
    figfmt = options.figfmt
    nFold = options.nFold
    desc = options.desc
    nClusterSample = options.nClusterSample
    
    if(options.verbose): 
        print dataset,nRowCluster,nColCluster,ccType,kernelType,beta,figfmt,nFold,nClusterSample
    
    dataPath = rootDir+dataset+dataDir
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    dataext = '.'+desc
    nCategory = len(catList)
    dim = descdim.get(desc)
    
    nSamplePerCategory = int(np.round(nClusterSample/nCategory))
    
    if(options.verbose): print 'collating cluster data...'
    clusterData = collateClusterData(dataPath,dataext,catList,nSamplePerCategory,dim)
    if(options.verbose): print 'coclustering...'
    ccData = cocluster.coclustWord(clusterData,dataset,nRowCluster,nColCluster,ccType)
    
    ccRow = np.array([int(i) for i in ccData[0].split()])
    ccCol = np.array([int(i) for i in ccData[1].split()])
    
    cctemp = np.zeros((clusterData.shape[0],nColCluster))
    codebook = np.zeros((nRowCluster,nColCluster))
    for i in np.arange(clusterData.shape[0]):
        for j in sorted(set(ccCol)):
            cctemp[i,j] = np.linalg.norm(clusterData[i,ccCol==j], 2)        ###############
#            cctemp[i,j] = np.mean(clusterData[i,ccCol==j])
    for i in sorted(set(ccRow)):
        codebook[i,:] = np.mean(cctemp[ccRow==i,:],0)
    
    if(options.verbose): print 'writing bof...'
    writebof(dataset,catList,codebook,ccCol,nRowCluster,desc,ccType,nRowCluster,nColCluster)
    
    perfMean = np.zeros(nCategory)
    perfStd = np.zeros(nCategory)
    for iCategory,catName in enumerate(catList):
#        catboffilepath = rootDir+dataset+bofDir+catName+'_cc'+bofext
        catboffilepath = rootDir+dataset+bofDir+catName+str(ccType)+str(nRowCluster)+str(nColCluster)+bofext
        catpos = np.genfromtxt(catboffilepath,dtype=np.int) # catpos
        catpos = catpos[:,:nColCluster+1]
        catpos[:,nColCluster] = 1
        for catname in catList:
            if(catname!=catName):
                firstvisit = True
#                catboffilepath = rootDir+dataset+bofDir+catname+'_cc'+bofext
                catboffilepath = rootDir+dataset+bofDir+catName+str(ccType)+str(nRowCluster)+str(nColCluster)+bofext
                if(firstvisit):
                    catneg = np.genfromtxt(catboffilepath,dtype=np.int)
                    firstvisit = False
                else : 
                    catneg = np.concatenate((catneg,np.genfromtxt(catboffilepath,dtype=np.int)),axis=0)
        nPos = catpos.shape[0]
        nNeg = catneg.shape[0]
        catneg = catneg[np.random.randint(0,nNeg,nPos),:] #catneg
        catneg = catneg[:,:nColCluster+1]
        catneg[:,nColCluster] = 0
        #combine positive and negative data
        catData = np.concatenate((catpos,catneg),axis=0)
        #shuffle the rows to aid in random selection of train and test
        np.random.shuffle(catData)
#        catPerf = classify.ccClassify(catData, kernelType, nFold, beta, nMetrics)
        catPerf = classify.knnClassify(catData, 10, nFold, beta, nMetrics)
        perfMean[iCategory] = np.mean(catPerf)
        perfStd[iCategory] = np.std(catPerf) 
    
    if(options.verbose):
        print perfMean
        print perfStd
    #plotresult.ccPlot(dataset, catList, perfMean, perfStd, figfmt, 'BoW', ccType)
    return [perfMean,perfStd]

def blWord():
    (options, args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
    
#    kernelType = options.kernelType
    nFold = options.nFold
    nCodeword = options.nCodeword   
    
    dataPath = rootDir+dataset+bofDir
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    if(nCodeword==1000):
        dataext = bofext
    else:
        dataext = str(nCodeword)+bofext
    nCategory = len(catList)
    
    perfMean = np.zeros(nCategory)
    perfStd = np.zeros(nCategory)
    
    for iCat,catname in enumerate(catList):
        print catname
        #read the category data which will positive
        fname = dataPath+catname+dataext
        catpos = np.genfromtxt(fname,dtype=np.int) # catpos
        catpos = catpos[:,:nCodeword]
        posLabel = np.ones((catpos.shape[0],1),dtype=np.int)
        catpos = np.concatenate((catpos,posLabel),axis=1)
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
        catneg = catneg[:,:nCodeword]
        negLabel = np.zeros((catneg.shape[0],1),dtype=np.int)
        catneg = np.concatenate((catneg,negLabel),axis=1)
        #combine positive and negative data
        data = np.concatenate((catpos,catneg),axis=0)
        #shuffle the rows to aid in random selection of train and test
        np.random.shuffle(data)
        
        X = data[:,:nCodeword]
        y = data[:,nCodeword]
        
#        clfParamList = {'kernel': kernelType, 'gamma': 1e-3, 'C': 1, 'degree':4, 'probability':True,'shrinking':True,'cache_size':1000}
#        classifier = SVC(**clfParamList)
        cv = StratifiedKFold(y, k=nFold)
        clfParamList = {'n_neighbors':10,'algorithm':'auto'}
        classifier = NeighborsClassifier(**clfParamList)
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
        
    if(options.verbose):
        print perfMean
        print perfStd
    
    return [perfMean,perfStd]

def ccnblWord():
    (options, args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
    ccType = options.ccType
    figfmt = options.figfmt
    nCodeword = options.nCodeword
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    # compute performance for the baseline implementation
    blResult = blWord()
    # compute performance for the co-clustering
    ccResult = ccWord()
    
    
    # compute the mean and std performance across all categories
    ccMean = np.mean(ccResult[0])
    ccStd = np.std(ccResult[0])
    blMean = np.mean(blResult[0])
    blStd = np.std(blResult[0])
    
    print "ccMean: %f" % ccMean
    print "ccStd: %f" % ccStd
    print "blMean: %f" % blMean
    print "blStd: %f" % blStd
    
    # store the result to the file using shelving in python
    resultfile = '{0}{1}{2}{3}{4}{5}'.format(rootDir,dataset,outDir,'ssrbcknnWord',ccType,str(nCodeword))
    result = shelve.open(resultfile)
    result['catList'] = catList
    result['ccResultMean'] = ccResult[0]
    result['ccResultStd'] = ccResult[1]
    result['blResultMean'] = blResult[0]
    result['blResultStd'] = blResult[1]
    result['ccMean'] = ccMean
    result['ccStd'] = ccStd
    result['blMean'] = blMean
    result['blStd'] = blStd
    result.close()
    
    # plot figure and store it to file
    fig1Title = '{0}{1}{2}'.format('ccblWord',ccType,str(nCodeword))
    outPath = '{0}{1}{2}{3}{4}{5}{6}{7}'.format(rootDir,dataset,outDir,'ssrbcknnWord',ccType,str(nCodeword),'.',figfmt)
    try:
        plotWordResult(ccResult,blResult,catList,dataset,outPath,fig1Title,figfmt)
    except:
        print 'unable to plot result, please experiment again'
    
if __name__ == '__main__':
    ccnblWord()
    pass