'''
Created on 10 Nov 2011
This program computes the image-word distribution for each category in each dataset
@author: ag00087
'''

#imports
import numpy as np
from optparse import OptionParser
import sys
from scipy.cluster.vq import kmeans2,vq

#acquire program arguments
parser = OptionParser()
parser.add_option('-d','--dataset',action='store',type='string',dest='dataset',default='VOC2006',metavar='dataset',help='visual dataset')
parser.add_option('-w','--nWord',action='store',type='int',dest='nWord',default=1024,metavar='nWord',help='number of words')
parser.add_option('-q','--quiet',action='store_false',dest='verbose',default=True)

#global paths
rootDir = '/vol/vssp/diplecs/ash/Data/'
dataDir = '/FeatureMatrix/'
outputDir = '/ImgWrdMat/'
auxOutputDir = '/UniversalCB/'

#global variables
catidfname = 'catidlist.txt' # list of categories in the dataset
cbext = '.ucb' # universal codebook
iwmext = '.iwm' # image word matrix
descriptorext = '.sift'
nDim = 128


def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.genfromtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.genfromtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap

def universalcodebook():
    #acquire program arguments
    (options, args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
    nWord = options.nWord
    
    #echo arguments if verbose
    if(options.verbose):
        print dataset,nWord
        
    #the dataPath given the dataset
    dataPath = rootDir+dataset+dataDir
    
    # set the sampleSize
    sampleSize = 2000
    #set the sampleSize based on the dataset
    if(dataset=='Caltech256'):
        sampleSize = 1000
    elif(dataset=='Caltech101'):
        sampleSize = 1500
    elif(dataset=='VOC2010'):
        sampleSize = 5000
    elif(dataset=='VOC2007'):
        sampleSize = 5000
    elif(dataset=='Scene15'):
        sampleSize = 10000
    elif(dataset=='VOC2006'):
        sampleSize = 15000
    else:
        sampleSize = 2000
    
    #acquire the category list
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    # the number of categories in category list
    nCategory = len(catList)
    
    #initialize the sampled data matrix to be clustered
    dataMat = np.empty([nCategory*sampleSize,nDim],dtype=np.int16)
    
    for iCategory,catName in enumerate(catList):
        if(options.verbose):print catName
        fileName = dataPath+catName+descriptorext
        if(options.verbose):print fileName
        fileData = np.loadtxt(fileName, dtype=np.int16, delimiter=' ')
        #randomly sample feature vectors
        nVectors = fileData.shape[0]
        sampleIdx = np.arange(nVectors)
        np.random.shuffle(sampleIdx)
        sampleIdx = sampleIdx[:sampleSize]
        categoryData = fileData[sampleIdx,:nDim]
        dataMat[(iCategory*sampleSize):((iCategory+1)*sampleSize),:] = categoryData
        pass
    
    # cluster the data matrix
    [codebook,_] = kmeans2(dataMat,nWord,iter=10,minit='random',missing='warn')
    
    #write the codebook to file
    ucbFileName = rootDir+dataset+auxOutputDir+dataset+str(nWord)+cbext
    try:
        np.savetxt(ucbFileName, codebook, fmt='%u', delimiter=' ')
    except:
        print 'unable to save codebook matrix to file %s' % (ucbFileName)
        
    for iCategory,catName in enumerate(catList):
        fileName = dataPath+catName+descriptorext
        fileData = np.loadtxt(fileName,dtype=np.int16,delimiter=' ')
        # find the images in the category
        imgIds = fileData[:,nDim:nDim+1]
        imgIds = np.unique(imgIds)
        # initialize output for category
        nImage = imgIds.size
        imgwrd = np.empty([nImage,nWord],dtype=np.int16)
        #iterate over each image
        for iImg,iIdx in enumerate(imgIds):
            imgData = fileData[fileData[:,nDim+1]==iIdx,:nDim]
            [code,_] = vq(imgData,codebook)
            #score each cluster
            for iWord in range(nWord):
                count = sum(code==iWord)
                imgwrd[iImg,iWord] = count
        
        #write the imgwrd matrix to file
        imwFileName = rootDir+dataset+outputDir+catName+str(nWord)+iwmext
        np.savetxt(imwFileName, imgwrd, fmt='%u', delimiter=' ')        
        pass
    
    pass

if __name__ == '__main__':
    universalcodebook()
    pass