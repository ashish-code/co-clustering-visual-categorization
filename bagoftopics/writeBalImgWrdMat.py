'''
Created on 16 Jan 2012
Project the ImgWrdMat data using PCA
The dictionary sizes are {32,64,128,256,512,1024,2048,4096,8192,16384}

@author: ag00087
'''

#imports
import numpy as np
from optparse import OptionParser
import sys


#acquire program arguments
parser = OptionParser()
parser.add_option('-d','--dataset',action='store',type='string',dest='dataset',default='VOC2006',metavar='dataset',help='visual dataset')
parser.add_option('-q','--quiet',action='store_false',dest='verbose',default=True)

#global paths
rootDir = '/vol/vssp/diplecs/ash/Data/'
iwmDir = '/ImgWrdMat/'
outputDir = '/PCA/'
universalcb = '/UniversalCB/'
imgidDir = '/ImgIds/'
universalworddictionary = '/UniversalWordDictionary/'
dims = [32,64,128,256,512,1024,2048,4096,8192,16384]

#global variables
catidfname = 'catidlist.txt' # list of categories in the dataset
cbext = '.ucb' # universal codebook
iwmext = '.iwm' # image word matrix
uwdext = '.uwd' # universal word dictionary
iidext = '.iid' # image id

def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.genfromtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.genfromtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap

def writeBalImgWrdMat(dim):
    (options, args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
#    method = options.method
    
    #acquire the category list
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    # the number of categories in category list
#    nCategory = len(catList)
    
    for catName in catList:
        print '%s : %d'%(catName,dim)
        catPosFileName = rootDir+dataset+iwmDir+catName+str(dim)+iwmext
        try:
            catPosData = np.loadtxt(catPosFileName, dtype=np.int, delimiter=' ')
        except:
            print 'unable to load %s'%catPosFileName
            break
        nPosImages = catPosData.shape[0]
        catNegFileName = rootDir+dataset+iwmDir+'NEG'+catName+str(dim)+iwmext
        try:
            catNegData = np.loadtxt(catNegFileName,dtype=np.int,delimiter=' ')
        except:
            print 'unable to load %s'%catNegFileName
            break
        nNegImages = catNegData.shape[0]
        catData = np.vstack((catPosData,catNegData))
        labels = np.vstack((np.ones((nPosImages,1),np.int),np.zeros((nNegImages,1),np.int)))
        balcatData = np.hstack((catData,labels))
        balCatDataFileName = rootDir+dataset+iwmDir+'BAL'+catName+str(dim)+iwmext
        np.savetxt(balCatDataFileName,balcatData,fmt='%d',delimiter=' ')
        print 'written %s'%balCatDataFileName
    pass

if __name__=='__main__':
    for dim in dims:
        writeBalImgWrdMat(dim)
        
            
    pass