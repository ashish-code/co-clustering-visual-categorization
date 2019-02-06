'''
Created on 15 Jan 2012
Write negative dataset for ImgWrdMatrices
@author: ag00087
'''

#import libraries
from optparse import OptionParser
import numpy as np
import sys
import os

#global parameters
rootDir = '/vol/vssp/diplecs/ash/Data/'
dataDir = '/ImgWrdMat/'

dataDim = {'lower':3}
outputDir = '/ImgWrdMat/'
catidfname = 'catidlist.txt'

dims = [32,64,128,256,512,1024,2048,4096,8192,16384]

parser = OptionParser()
parser.add_option('-d','--dataset',type='string',metavar='dataset',dest='dataset',default='VOC2006',help='the dataset')


def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.loadtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.loadtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap    

def genNegativeData(lowerDim):
    (options,args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    nCategory = len(catList)
    dataDim['lower']= lowerDim
    dataExt = str(dataDim.get('lower'))+'.iwm'
    for catName in (catList):
        catNegFilePath = rootDir+dataset+outputDir+'NEG'+catName+dataExt
        if os.path.exists(catNegFilePath): continue
        print catName
        catFilePath = rootDir+dataset+dataDir+catName+dataExt
        if not os.path.exists(catFilePath): continue
        catData = np.genfromtxt(catFilePath,dtype=np.int)
        
        #catId = np.genfromtxt(catFilePath,dtype=np.int,usecols=np.arange(dim+1,dim+2))
        nImage = catData.shape[0]
        print 'nImage: %d'%(nImage)
        nSampleImagePerCategory = np.round(nImage/(nCategory-1))
        #at least one image is selected from the dataset
        if nSampleImagePerCategory < 1:
            nSampleImagePerCategory = 1
        print 'nSampleImagePerCategory: %d'%(nSampleImagePerCategory)
        
        createSwitch = True
        for nameCat in (catList):
            if nameCat != catName:
                print nameCat
                catFilePath = rootDir+dataset+dataDir+nameCat+dataExt
                catData = np.genfromtxt(catFilePath,dtype=np.int)
                numimg = catData.shape[0]
                if (numimg > nSampleImagePerCategory):
                    catSampleData = catData[:nSampleImagePerCategory,:]
                else:
                    catSampleData = catData
                               
                
                
                if createSwitch:
                    createSwitch = False
                    catNegData = catSampleData
                else:
                    catNegData = np.concatenate((catNegData,catSampleData),axis=0)
            
            pass
        # write the category negative to file
        
        print 'writing %s'%(catNegFilePath)
        np.savetxt(catNegFilePath, catNegData, fmt='%d', delimiter=' ')
        pass
    pass

if __name__=='__main__':
    for dim in dims:
        genNegativeData(dim)
    pass