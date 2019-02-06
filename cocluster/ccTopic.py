'''
Created on 27 Jul 2011

@author: ag00087
'''
import numpy as np
import sys
import cocluster,plotresult,classify

from optparse import OptionParser

parser = OptionParser()
parser.add_option('-d','--dataset',type='string',metavar='dataset',dest='dataset',default='VOC2006',help='the dataset')
parser.add_option('-r','--rowClusters',type='int',metavar='nRowCluster',dest='nRowCluster',default=1,help='number of row clusters')
parser.add_option('-c','--colClusters',type='int',metavar='nColCluster',dest='nColCluster',default=100,help='number of column clusters')
parser.add_option('-t','-s','--ccType','--ccScheme',type='string',metavar='ccType',dest='ccType',default='e',help='type of co-clustering scheme: e,i,r')
parser.add_option('-k','--kernel',type='string',metavar='kernelType',dest='kernelType',default='linear',help='type of svm kernel, linear,rbf,poly,sigmoid')
parser.add_option('-x','--nFoldXVal',type='int',metavar='nFold',dest='nFold',default=2,help='number of stratified cross validation iterations')
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
        if(options.verbose): print 'coclustering...'
        ccData = cocluster.coclust(bofData, dataset, nRowCluster, nColCluster, ccType)
                
        ccCol = np.array([int(i) for i in ccData[1].split()])
        tempCC = np.zeros((bofData.shape[0],nColCluster))
        for i in np.arange(bofData.shape[0]):
            for j in sorted(set(ccCol)):
                tempCC[i,j] = np.sum(bofData[i,ccCol==j])
     
        botData = np.vstack((tempCC.T,bofData[:,-1])).T
        
        if(options.verbose): print 'classifying...'
        catPerf = classify.ccClassify(botData, kernelType, nFold, beta, nMetrics)
        perfMean[iCategory] = np.mean(catPerf)
        perfStd[iCategory] = np.std(catPerf) 

    if(options.verbose):
        print perfMean
        print perfStd
    plotresult.ccPlot(dataset, catList, perfMean, perfStd, figfmt, 'BoT', ccType)

if __name__ == '__main__':
    ccTopic()