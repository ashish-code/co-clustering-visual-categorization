'''
Created on 11 Nov 2011
Program to create a imgWrd matrix for each category of each dataset
this program will utilize scikits.learn
@author: ag00087
'''


#imports
import numpy as np
from optparse import OptionParser
import sys
from sklearn.cluster import MiniBatchKMeans
from time import time

#acquire program arguments
parser = OptionParser()
parser.add_option('-d','--dataset',action='store',type='string',dest='dataset',default='VOC2006',metavar='dataset',help='visual dataset')
parser.add_option('-w','--nWord',action='store',type='int',dest='nWord',default=1024,metavar='nWord',help='number of words')
parser.add_option('-q','--quiet',action='store_false',dest='verbose',default=True)

#global paths
rootDir = '/vol/vssp/diplecs/ash/Data/'
dataDir = '/FeatureMatrix/'
outputPath = '/vol/vssp/diplecs/ash/conference/cvpr2012/nmf/data/'
outputDir = '/'
auxOutputDir = '/'

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

def imgWrd():
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
    sampleSize = 1000
    #set the sampleSize based on the dataset
    if(dataset=='Caltech256'):
        sampleSize = 500
    elif(dataset=='Caltech101'):
        sampleSize = 800
    elif(dataset=='VOC2010'):
        sampleSize = 2000
    elif(dataset=='VOC2007'):
        sampleSize = 2000
    elif(dataset=='Scene15'):
        sampleSize = 5000
    elif(dataset=='VOC2006'):
        sampleSize = 5000
    else:
        sampleSize = 1000
    
    #acquire the category list
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    # the number of categories in category list
    nCategory = len(catList)
    
    #initialise the sampled data matrix to be clustered
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
    
    #compute the codebook using sklearn.cluster.MiniBatchKMeans
    mbkm = MiniBatchKMeans(init="k-means++", k=nWord, max_iter=10, random_state=None,
                       chunk_size=1000)
    print "Clustering sparse data with %s" % str(mbkm)
    t0 = time()
    mbkm.fit(dataMat)
    print "done in %0.3fs" % (time() - t0)
    
    codebook = mbkm.cluster_centers_
    
    #write the codebook to file
    ucbFileName = outputPath+dataset+auxOutputDir+dataset+str(nWord)+cbext
    try:
        np.savetxt(ucbFileName, codebook, fmt='%u', delimiter=' ')
    except:
        print 'unable to save codebook matrix to file %s' % (ucbFileName)

    if(options.verbose): print 'predicting label...'
    for iCategory,catName in enumerate(catList):
        fileName = dataPath+catName+descriptorext
        fileData = np.loadtxt(fileName,dtype=np.int16,delimiter=' ')
        if(options.verbose): print fileName
        # find the images in the category
        imgIds = fileData[:,nDim:nDim+1]
        imgIds = np.unique(imgIds)
        # initialise output for category
        nImage = imgIds.size
        imgwrd = np.empty([nImage,nWord],dtype=np.int16)
        #iterate over each image
        for iImg,iIdx in enumerate(imgIds):
            imgData = fileData[fileData[:,nDim+1]==iIdx,:nDim]
            code = mbkm.predict(imgData)
            #score each cluster
            for iWord in range(nWord):
                count = sum(code==iWord)
                imgwrd[iImg,iWord] = count
        
        #write the imgwrd matrix to file
        imwFileName = outputPath+dataset+outputDir+catName+str(nWord)+iwmext
        try:
            np.savetxt(imwFileName, imgwrd, fmt='%u', delimiter=' ')
            if(options.verbose): print 'writing file: %s'%(imwFileName)
        except:
            print 'unable to write to file %s'%(imwFileName)            
        pass

if __name__ == '__main__':
    imgWrd()
    pass