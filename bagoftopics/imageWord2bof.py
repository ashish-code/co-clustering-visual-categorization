'''
Created on 12 Apr 2012
modified for bof directory output
@author: ag00087
'''
'''
Created on 21 Nov 2011
original version
@author: ag00087
'''

#imports
import numpy as np
from optparse import OptionParser
import sys
from scipy.cluster.vq import kmeans2,vq
from sklearn.cluster import MiniBatchKMeans

#acquire program arguments
parser = OptionParser()
parser.add_option('-d','--dataset',action='store',type='string',dest='dataset',default='VOC2006',metavar='dataset',help='visual dataset')
parser.add_option('-w','--nWord',action='store',type='int',dest='nWord',default=16384,metavar='nWord',help='number of words')
parser.add_option('-q','--quiet',action='store_false',dest='verbose',default=True)

#global paths
rootDir = '/vol/vssp/diplecs/ash/Data/'
dataDir = '/FeatureMatrix/'
#outputDir = '/ImgWrdMat/'
outputDir = '/BOF/'
universalcb = '/UniversalCB/'
imgidDir = '/ImgIds/'
universalworddictionary = '/UniversalWordDictionary/'
itmlClusterDir = '/itmlCluster/'

#global variables
catidfname = 'catidlist.txt' # list of categories in the dataset
cbext = '.ucb' # universal codebook
#iwmext = '.iwm' # image word matrix
iwmext = '.bof'
descriptorext = '.sift'
uwdext = '.uwd' # universal word dictionary
iidext = '.iid' # image id
nDim = 128


def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.genfromtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.genfromtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap

def universalcodebookWordDictionary():
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
        sampleSize = 500
    elif(dataset=='VOC2010'):
        sampleSize = 1000
    elif(dataset=='VOC2007'):
        sampleSize = 1000
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
        categoryData = fileData[sampleIdx,2:nDim+2]
        dataMat[(iCategory*sampleSize):((iCategory+1)*sampleSize),:] = categoryData
        pass
    mbkm = MiniBatchKMeans(init="k-means++", k=nWord, max_iter=1, random_state=None,
                       chunk_size=10)
    # cluster the data matrix
    [codebook,_] = kmeans2(dataMat,nWord,iter=20,minit='random',missing='warn')
    mbkm.cluster_centers_ = codebook
    #write the code-book to file
    ucbFileName = rootDir+dataset+universalcb+dataset+str(nWord)+cbext
    try:
        np.savetxt(ucbFileName, codebook, fmt='%u', delimiter=' ')
    except:
        print 'unable to save codebook matrix to file %s' % (ucbFileName)
        
    for iCategory,catName in enumerate(catList):
        fileName = dataPath+catName+descriptorext
        fileData = np.loadtxt(fileName,dtype=np.int16,delimiter=' ')
        # find the images in the category
        imgIds = fileData[:,-2]
        imgIdsFileName = rootDir+dataset+imgidDir+catName+str(nWord)+iidext
        try:
            np.savetxt(imgIdsFileName, imgIds, fmt='%d', delimiter=' ')
        except:
            print 'unable to write %s'%(imgIdsFileName)
        imgIds = np.unique(imgIds)
        # initialize output for category
        nImage = imgIds.size
        imgwrd = np.empty([nImage,nWord],dtype=np.int16)
        #iterate over each image
        uwdcode = None
        for iImg,iIdx in enumerate(imgIds):
            imgData = fileData[fileData[:,-2]==iIdx,2:nDim+2]
            [code,_] = vq(imgData,codebook)
            
            try:
                uwdcodeimg = mbkm.predict(imgData)
            except:
                print 'unable to compute universal word dictionary'
            if(uwdcode == None):
                uwdcode = uwdcodeimg
            else:
                uwdcode = np.append(uwdcode,uwdcodeimg)
            #score each cluster
            for iWord in range(nWord):
                count = sum(code==iWord)
                imgwrd[iImg,iWord] = count
        uwdFileName = rootDir+dataset+universalworddictionary+catName+str(nWord)+uwdext
        try:
            np.savetxt(uwdFileName, uwdcode, fmt='%d', delimiter=' ')
            print 'writing %s'%(uwdFileName)
        except:
            print 'unable to write %s'%(uwdFileName)
        #write the imgwrd matrix to file
        imwFileName = rootDir+dataset+outputDir+catName+str(nWord)+iwmext
        np.savetxt(imwFileName, imgwrd, fmt='%u', delimiter=' ')
        print 'writing %s'%(imwFileName)        
        pass
    
    pass

if __name__ == '__main__':
    universalcodebookWordDictionary()
    pass