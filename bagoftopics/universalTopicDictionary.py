'''
Created on 7 Dec 2011

@author: ag00087
'''

# import libraries
import numpy as np
from optparse import OptionParser
import sys
import os
import time
import shutil

# parse options
parser = OptionParser()
parser.add_option('-d','--dataset',action='store',type='string',dest='dataset',metavar='dataset',help='visual dataset')
parser.add_option('-w','--nCodeword',action='store',type='int',dest='nCodeword',default=1024,metavar='nCodeword',help='number of code words: 32,64,128,256,512,1024,2048,4096,8192,16384')
parser.add_option('-t','--nTopic',action='store',type='int',dest='nTopic',metavar='nTopic',default=128,help='number of topic words: 8,16,32,64,128,256,512,1024')
parser.add_option('-s','--scheme',action='store',type='string',dest='ccType',metavar='ccType',default='i',help='ccType')
parser.add_option('-q','--quiet',action='store_false',dest='verbose',default=True)
parser.add_option('-r','--rowClusters',type='int',metavar='nRowCluster',dest='nRowCluster',default=10,help='number of row clusters')


#configure data paths
rootDir = '/vol/vssp/diplecs/ash/Data/'
imgWrdDir = '/ImgWrdMat/'
ucbDir = '/UniversalCB/'
utdDir = '/UniversalTopicDictionary/'
tempDir = 'Temp'

# global variables
catidfname = 'catidlist.txt'
ucbext = '.ucb'
imgWrdext = '.iwm'
utdext = '.utd'

def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.genfromtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.genfromtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap

def ccUniversalTopicDictionary():
    #acquire program agruments
    (options, args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
    nRowCluster = options.nRowCluster
    nTopic = options.nTopic
    ccType = options.ccType
    nCodeword = options.nCodeword
    
    
    #echo arguments    
    if(options.verbose): print options
        
    #configure data path and other parameters
    dataPath = rootDir+dataset+imgWrdDir
    resultPath = rootDir+dataset+utdDir+dataset
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    dataext = str(nCodeword)+imgWrdext
    resultext = str(nCodeword)+utdext
    
    
    # initialise empty iwm matrix and append each category to it
    iwmData = None
    for catName in catList:
        iwmFileName = dataPath+catName+dataext
        iwmCatData = np.loadtxt(iwmFileName,dtype=np.int16,delimiter=' ')
        #stack the category data to the existing data-set data
        if iwmData == None:
            iwmData = iwmCatData
        else:
            iwmData = np.concatenate((iwmData,iwmCatData),axis=0)
        pass
    
    if(options.verbose): print 'co-clustering...'
            
    ccData = coclust(iwmData, dataset, nRowCluster, nTopic, ccType)
    # the indices of co-clusters columns
    ccCol = np.array([int(i) for i in ccData[1].split()])
    
    resultFileName = resultPath+resultext
    np.savetxt(resultFileName, ccCol, fmt='%d', delimiter=' ')

def UniversalTopicDictionary(nCodeword,nTopic,dataset):
    nRowCluster = 10    
    ccType = 'i'   
            
    #configure data path and other parameters
    dataPath = rootDir+dataset+imgWrdDir
    resultPath = rootDir+dataset+utdDir+dataset
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    dataext = str(nCodeword)+imgWrdext
    resultext = str(nCodeword)+utdext
        
    # initialise empty iwm matrix and append each category to it
    iwmData = None
    for catName in catList:
        iwmFileName = dataPath+catName+dataext
        iwmCatData = np.loadtxt(iwmFileName,dtype=np.int16,delimiter=' ')
        #stack the category data to the existing data-set data
        if iwmData == None:
            iwmData = iwmCatData
        else:
            iwmData = np.concatenate((iwmData,iwmCatData),axis=0)
        pass
            
    ccData = coclust(iwmData, dataset, nRowCluster, nTopic, ccType)
    # the indices of co-clusters columns
    ccCol = np.array([int(i) for i in ccData[1].split()])
    
    resultFileName = resultPath+resultext
    np.savetxt(resultFileName, ccCol, fmt='%d', delimiter=' ')


def coclust(data,dataset,nRowCluster=1,nColCluster=1,ccType='i'):
    
    tempTimeDir = str(int(time.time()))
    tempPath = rootDir+tempDir+'/'+tempTimeDir+'/'
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    tempdataPath = tempPath+'tempdata'
    tempdataDimPath = tempPath+'tempdata_dim'
    tempCCFilePath = tempPath+'tempCCFile.txt'
    np.savetxt(tempdataPath,data[:,:-1],fmt='%d',delimiter=' ')
    np.savetxt(tempdataDimPath,data[:,:-1].shape,fmt='%d',delimiter=' ')
    
    cmdPath = '/vol/vssp/diplecs/ash/code/cocluster/'
    cmd = 'cocluster-linux'
    args = ' -A %s -R %d -C %d -I d s %s -O c s 0 o %s' % (ccType,nRowCluster,nColCluster,tempdataPath,tempCCFilePath)
    cwd = os.getcwd()
    os.chdir(cmdPath)
    os.system(cmd + args)
    os.chdir(cwd)
    
    tempCCFile = open(tempCCFilePath,'r')
    tempFileData = tempCCFile.readlines()
    tempCCFile.close()
    
    # delete temp data
    shutil.rmtree(tempPath)
    
    return tempFileData   

if __name__ == '__main__':
    ccUniversalTopicDictionary()
    pass