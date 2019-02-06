'''
Created on 18 Nov 2011

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
imgidDir = '/ImgIds/'
universalworddictionary = '/UniversalWordDictionary/'
universalcb = '/UniversalCB/'
#global variables
catidfname = 'catidlist.txt' # list of categories in the dataset
cbext = '.ucb' # universal codebook
iwmext = '.iwm' # image word matrix
uwdext = '.uwd' # universal word dictionary
iidext = '.iid' # image id
descriptorext = '.sift'
nDim = 128


def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.genfromtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.genfromtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap

def universalWordDictionary():
    #acquire program arguments
    (options, args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
    nWord = options.nWord
    
    #echo arguments if verbose
    if(options.verbose):
        print dataset,nWord
        
    #the dataPath given the dataset
    dataPath = rootDir+dataset+dataDir
    #acquire the category list
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    # the number of categories in category list
    mbkm = MiniBatchKMeans(init="k-means++", k=nWord, max_iter=1, random_state=None,
                       chunk_size=10)
    
    #load the codebook and assigning to cluster centers of the minibatchkmeans clustering object
    ucbFileName = rootDir+dataset+universalcb+dataset+str(nWord)+cbext
    try:
        codebook = np.loadtxt(ucbFileName, dtype=np.int, delimiter=' ')
    except:
        print 'unable to read codebook file %s' % (ucbFileName)
    #assign the codebook to the clustering object centroids
    mbkm.cluster_centers_ = codebook
    
    if(options.verbose): print 'predicting label...'
    for catName in catList:
        t0 = time()
        fileName = dataPath+catName+descriptorext
        fileData = np.loadtxt(fileName,dtype=np.int16,delimiter=' ')
        if(options.verbose): print fileName
        imgIds = fileData[:,nDim:nDim+1]
        imgIdsFileName = rootDir+dataset+imgidDir+catName+str(nWord)+iidext
        try:
            np.savetxt(imgIdsFileName, imgIds, fmt='%d', delimiter=' ')
        except:
            print 'unable to write %s'%(imgIdsFileName)
        #iterate over each image
        imgData = fileData[:,:nDim]
        code = mbkm.predict(imgData)
        uwdFileName = rootDir+dataset+universalworddictionary+catName+str(nWord)+uwdext
        try:
            np.savetxt(uwdFileName, code, fmt='%d', delimiter=' ')
        except:
            print 'unable to write %s'%(uwdFileName)
        print "done in %0.3fs" % (time() - t0)

if __name__ == '__main__':
    universalWordDictionary()
    pass