'''
Created on 17 Aug 2011
comparison of co-clustering and baseline bag-of-features
@author: ag00087
'''
import numpy as np
from optparse import OptionParser
import sys
import cocluster, classify
#from ccnblplt import plotClassifierResult
import shelve
from ccnblplt import plotKNNresult

parser = OptionParser()
parser.add_option('-d','--dataset',action='store',type='string',dest='dataset',metavar='dataset',help='visual dataset')
parser.add_option('-w','--nCodeword',action='store',type='int',dest='nCodeword',default=10000,metavar='nCodeword',help='number of code words')
parser.add_option('-t','--nTopic',action='store',type='int',dest='nTopic',metavar='nTopic',default=1000,help='number of topic words')
parser.add_option('-s','--scheme',action='store',type='string',dest='ccType',metavar='ccType',default='r',help='ccType')
parser.add_option('-q','--quiet',action='store_false',dest='verbose',default=True)
parser.add_option('-x','--nFoldXVal',type='int',metavar='nFold',dest='nFold',default=4,help='number of stratified cross validation iterations')
parser.add_option('-p','--figformat',type='string',metavar='figfmt',dest='figfmt',default='svg',help='type of output graph image, png, svg, jpg')
parser.add_option('-r','--rowClusters',type='int',metavar='nRowCluster',dest='nRowCluster',default=10,help='number of row clusters')
parser.add_option('-b','--beta',type='float',metavar='beta',dest='beta',default=1.0,help='f-beta value, default is 1.0')
parser.add_option('-n','--neighbors',type='int',metavar='neighbors',dest='neighbors',default=10,help='number of neighbours for knn classifier')
parser.add_option('-k','--kernelType',type='string',metavar='kernelType',dest='kernelType',default='rbf',help='svm kernel type')

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
nClassifier = 2

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
    nTopic = options.nTopic
    ccType = options.ccType
    kernelType = options.kernelType
    nFold = options.nFold
    nCodeword = options.nCodeword
    beta = options.beta
        
    if(options.verbose): 
        print dataset,nRowCluster,nTopic,ccType,kernelType,beta,nFold,nCodeword
        print options
    
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
        catneg = catneg[np.random.randint(0,nNeg,nPos),:] 
        catneg = catneg[:,:nCodeword+1]
        catneg[:,nCodeword] = 0
        #combine positive and negative data
        bofData = np.concatenate((catpos,catneg),axis=0)
        if(options.verbose): 
            print 'co-clustering...'
            
        ccData = cocluster.coclust(bofData, dataset, nRowCluster, nTopic, ccType)
                
        ccCol = np.array([int(i) for i in ccData[1].split()])
        tempCC = np.zeros((bofData.shape[0],nTopic))
        for i in np.arange(bofData.shape[0]):
            for j in sorted(set(ccCol)):
                tempCC[i,j] = np.sum(bofData[i,ccCol==j])
     
        botData = np.vstack((tempCC.T,bofData[:,-1])).T
        
        if(options.verbose): 
            print 'classifying...'
        
        
        #catPerfSVM = classify.ccClassify(botData, kernelType, nFold, beta, nMetrics)
        catPerfKNN = classify.knnClassify(botData, 10, nFold, beta, nMetrics)
        #perfMean[iCategory,0] = np.mean(catPerfSVM)
        #perfStd[iCategory,0] = np.std(catPerfSVM)
        perfMean[iCategory] = np.mean(catPerfKNN)
        perfStd[iCategory] = np.std(catPerfKNN) 

    if(options.verbose):
        print perfMean
        print perfStd
    return perfMean


def blTopic():
    (options, args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
    neighbors = options.neighbors
#    kernelType = options.kernelType
    nFold = options.nFold
    nTopic = options.nTopic
    beta = options.beta
    dataPath = rootDir+dataset+bofDir
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    #remove if bof of 1000 words is computed for all categories
    if(nTopic==1000):
        dataext = bofext
    else:
        dataext = str(nTopic)+bofext
    
    nCategory = len(catList)
    
    perfMean = np.zeros(nCategory)
    perfStd = np.zeros(nCategory)
    
    for iCategory,catname in enumerate(catList):
        print catname
        #read the category data which will positive
        fname = dataPath+catname+dataext
        catpos = np.genfromtxt(fname,dtype=np.int) 
        catpos = catpos[:,:nTopic+1]
        catpos[:,nTopic] = 1
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
        catneg = catneg[np.random.randint(0,nNeg,nPos),:] 
        catneg = catneg[:,:nTopic+1]
        catneg[:,nTopic] = 0
        #combine positive and negative data
        botData = np.concatenate((catpos,catneg),axis=0)
        #shuffle the rows to aid in random selection of train and test
        np.random.shuffle(botData)
        
        if(options.verbose): 
            print 'classifying...'
        
       
        #catPerfSVM = classify.ccClassify(botData, kernelType, nFold, beta, nMetrics)
        catPerfKNN = classify.knnClassify(botData, neighbors, nFold, beta, nMetrics)
        #perfMean[iCategory,0] = np.mean(catPerfSVM)
        #perfStd[iCategory,0] = np.std(catPerfSVM)
        perfMean[iCategory] = np.mean(catPerfKNN)
        perfStd[iCategory] = np.std(catPerfKNN) 
    
    if(options.verbose):
        print perfMean
        print perfStd            
    return perfMean
    

def ccnblTopic():
    (options, args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
    ccType = options.ccType
    figfmt = options.figfmt
    nTopic = options.nTopic
    nCodeword = options.nCodeword
    
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    
    ccResult = ccTopic()
    blResult = blTopic()
    #store the results to file
    resultfile = '{0}{1}{2}{3}{4}{5}{6}'.format(rootDir,dataset,outDir,'ccblTopic',ccType,str(nTopic),str(nCodeword))
    result = shelve.open(resultfile)
    result['catList'] = catList
    result['ccResultMean'] = ccResult
#    result['ccResultStd'] = ccResult
    result['blResultMean'] = blResult
#    result['blResultStd'] = blResult[1]
    result.close()
    fig1Title = '{0}{1}{2}{3}'.format('ccblTopic',ccType,str(nTopic),str(nCodeword))
    #plotresult(dataset,catList,ccResult,blResult,figfmt,fig1Title,ccType)
    outPath = '{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(rootDir,dataset,outDir,'ccblTopic',ccType,str(nTopic),str(nCodeword),'.',figfmt)
    try:
        plotKNNresult(ccResult,blResult,catList,dataset,outPath,fig1Title,figfmt)
    except:
        print 'unable to plot result'
    
    
if __name__ == '__main__':
    ccnblTopic()