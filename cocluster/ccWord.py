'''
Created on 27 Jul 2011

@author: ag00087
'''

import numpy as np
import sys
import cocluster,plotresult,classify
from optparse import OptionParser
from scipy.cluster.vq import vq

rootDir = '/vol/vssp/diplecs/ash/Data/'
dataDir = '/FeatureMatrix/'
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
descdim = {'sift':128}

parser = OptionParser()
parser.add_option('-d','--dataset',type='string',metavar='dataset',dest='dataset',default='VOC2006',help='the dataset')
parser.add_option('-r','--rowClusters',type='int',metavar='nRowCluster',dest='nRowCluster',default=100,help='number of row clusters')
parser.add_option('-c','--colClusters',type='int',metavar='nColCluster',dest='nColCluster',default=100,help='number of column clusters')
parser.add_option('-t','--ccType',type='string',metavar='ccType',dest='ccType',default='e',help='type of co-clustering scheme: e,i,r')
parser.add_option('-k','--kernel',type='string',metavar='kernelType',dest='kernelType',default='linear',help='type of svm kernel, linear,rbf,poly,sigmoid')
parser.add_option('-x','--nFoldXVal',type='int',metavar='nFold',dest='nFold',default=2,help='number of stratified cross validation iterations')
parser.add_option('-p','--figformat',type='string',metavar='figfmt',dest='figfmt',default='png',help='type of output graph image, png, svg, jpg')
parser.add_option('-b','--beta',type='float',metavar='beta',dest='beta',default=1.0,help='f-beta value, default is 1.0')
parser.add_option("-q", "--quiet", action="store_false", dest="verbose",default=True,help='verbosity')
parser.add_option('-f','--featureDescriptor',type='string',metavar='desc',dest='desc',default='sift',help='feature descriptor')
parser.add_option('-s','--nclusterSample',type='int',metavar='clusterSampleSize',dest='nClusterSample',default=10000,help='number of samples for clustering')

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
        catData = np.loadtxt(catFilePath,dtype=np.int8,usecols=np.arange(2,dim+2))
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

def writebof(dataset,catList,codebook,colCluster,nRowCluster,desc):
    for catName in catList:
        dataext = '.'+desc
        dim = descdim.get(desc)
        catFilePath = rootDir+dataset+dataDir+catName+dataext
        catData = np.genfromtxt(catFilePath,dtype=np.int8,usecols=np.arange(2,dim+2))
        catImgId = np.genfromtxt(catFilePath,dtype=np.int,usecols=np.arange(dim+2,dim+3))
        catId = np.genfromtxt(catFilePath,dtype=np.int,usecols=np.arange(dim+3,dim+4))[0]
        ImgId = np.unique(catImgId)
        nCol = codebook.shape[1]
        #collate the columns
        catD = np.zeros((catData.shape[0],nCol))
        for i in np.arange(catData.shape[0]):
            for j in sorted(set(colCluster)):
                catD[i,j] = np.linalg.norm(catData[i,colCluster==j], 2)
        
        # vq the data
        print catD.shape
        print codebook.shape
        [catLabel,catDist] = vq(catD,codebook) #@UnusedVariable
        # write the bag-of-features to file
        catboffilepath = rootDir+dataset+bofDir+catName+'_cc'+bofext
        catboffile = open(catboffilepath,'w')
        imgcount=0
        for imgid in ImgId:
            imgLabel = catLabel[catImgId==imgid]
            [hist,edges] = np.histogram(imgLabel,nRowCluster) #@UnusedVariable
            if imgcount==0:
                dataout = np.hstack((hist.T,imgid,catId))
            else:
                dataout = np.vstack((dataout,np.hstack((hist.T,imgid,catId))))
            imgcount+=1
        np.savetxt(catboffile, dataout, fmt='%d', delimiter=' ', )
        catboffile.close()

def ccWord():
    (options,args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
    nRowCluster = options.nRowCluster
    nColCluster = options.nColCluster
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
    ccData = cocluster.coclust(clusterData,dataset,nRowCluster,nColCluster,ccType)
    
    ccRow = np.array([int(i) for i in ccData[0].split()])
    ccCol = np.array([int(i) for i in ccData[1].split()])
    
    cctemp = np.zeros((clusterData.shape[0],nColCluster))
    codebook = np.zeros((nRowCluster,nColCluster))
    for i in np.arange(clusterData.shape[0]):
        for j in sorted(set(ccCol)):
            cctemp[i,j] = np.linalg.norm(clusterData[i,ccCol==j], 2)
    for i in sorted(set(ccRow)):
        codebook[i,:] = np.mean(cctemp[ccRow==i,:],0)
    
    if(options.verbose): print 'writing bof...'
    writebof(dataset,catList,codebook,ccCol,nRowCluster,desc)
    
    perfMean = np.zeros(nCategory)
    perfStd = np.zeros(nCategory)
    for iCategory,catName in enumerate(catList):
        catboffilepath = rootDir+dataset+bofDir+catName+'_cc'+bofext
        catpos = np.genfromtxt(catboffilepath,dtype=np.int) # catpos
        catpos = catpos[:,:nColCluster+1]
        catpos[:,nColCluster] = 1
        for catname in catList:
            if(catname!=catName):
                firstvisit = True
                catboffilepath = rootDir+dataset+bofDir+catname+'_cc'+bofext
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
        catPerf = classify.ccClassify(catData, kernelType, nFold, beta, nMetrics)
        perfMean[iCategory] = np.mean(catPerf)
        perfStd[iCategory] = np.std(catPerf) 
    
    if(options.verbose):
        print perfMean
        print perfStd
    plotresult.ccPlot(dataset, catList, perfMean, perfStd, figfmt, 'BoW', ccType)
        
if __name__ == '__main__':
    ccWord()