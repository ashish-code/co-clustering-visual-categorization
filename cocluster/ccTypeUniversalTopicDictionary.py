'''
Created on 17 Nov 2011

@author: ag00087
'''

# import libraries
import numpy as np
from optparse import OptionParser
import sys
import os
from cocluster import coclust

# parse options
parser = OptionParser()
parser.add_option('-d','--dataset',action='store',type='string',dest='dataset',metavar='dataset',help='visual dataset')
parser.add_option('-w','--nCodeword',action='store',type='int',dest='nCodeword',default=1024,metavar='nCodeword',help='number of code words: 1024,2048,4096,8192,16384')
parser.add_option('-t','--nTopic',action='store',type='int',dest='nTopic',metavar='nTopic',default=128,help='number of topic words: 32,64,128,256,512')
parser.add_option('-s','--scheme',action='store',type='string',dest='ccType',metavar='ccType',default='e',help='ccType')
parser.add_option('-q','--quiet',action='store_false',dest='verbose',default=True)
parser.add_option('-x','--nFoldXVal',type='int',metavar='nFold',dest='nFold',default=10,help='number of stratified cross validation iterations')
parser.add_option('-p','--figformat',type='string',metavar='figfmt',dest='figfmt',default='pdf',help='type of output graph image, png, svg, jpg')
parser.add_option('-r','--rowClusters',type='int',metavar='nRowCluster',dest='nRowCluster',default=10,help='number of row clusters')
parser.add_option('-b','--beta',type='float',metavar='beta',dest='beta',default=1.0,help='f-beta value, default is 1.0')
parser.add_option('-n','--neighbors',type='int',metavar='neighbors',dest='neighbors',default=10,help='number of neighbours for knn classifier')
parser.add_option('-k','--kernelType',type='string',metavar='kernelType',dest='kernelType',default='rbf',help='svm kernel type')

#configure data paths
rootDir = '/vol/vssp/diplecs/ash/Data/'
imgWrdDir = '/ImgWrdMat/'
tempDir = '/Temp/'
ucbDir = '/UniversalCB/'
utdDir = '/UniversalTopicDictionary/'


# global variables
catidfname = 'catidlist.txt'
ucbext = '.ucb'
imgWrdext = '.iwm'
utdext = '.utd'

#global parametersets
nWords = [1024,2048,4096,8192,16384]
nTopics = [32,64,128,256,512]

def getCatMap(dataset):
    catidfpath = rootDir+dataset+'/'+catidfname
    catnames = np.genfromtxt(catidfpath,delimiter=',',dtype='|S32',usecols=[0])
    catnum = np.genfromtxt(catidfpath,delimiter=',',dtype=np.int,usecols=[1])
    catmap = dict(zip(catnames,catnum))
    return catmap

def ccUniversalTopicDictionary(wordn,topicn):
    #acquire program agruments
    (options, args) = parser.parse_args(sys.argv[1:]) #@UnusedVariable
    dataset = options.dataset
    nRowCluster = options.nRowCluster
    
    ccType = options.ccType
    #nCodeword = options.nCodeword
    #nTopic = options.nTopic
    nCodeword = wordn
    nTopic = topicn
    
    #echo arguments    
    if(options.verbose): 
        print dataset,nRowCluster,nTopic,ccType,nCodeword
        print options
        
    #configure data path and other parameters
    dataPath = rootDir+dataset+imgWrdDir
    resultPath = rootDir+dataset+utdDir+ccType+dataset
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    dataext = str(nCodeword)+imgWrdext
    resultext = str(nCodeword)+str(nTopic)+utdext
    resultFileName = resultPath+resultext
    if(os.path.exists(resultFileName)): 
        print '%s already written' % (resultFileName)
        return
    #flag if incomplete data
    incompleteData = False
    
    for catName in catList:
        iwmFileName = dataPath+catName+dataext
        if(os.path.exists(iwmFileName)==False):
            incompleteData = True
            print '%s missing in %s,%d' %(catName,dataset,wordn)
            return
        else:
            pass
    
    # initialise empty iwm matrix and append each category to it
    iwmData = None
    for catName in catList:
        iwmFileName = dataPath+catName+dataext
        try:
            iwmCatData = np.loadtxt(iwmFileName,dtype=np.int16,delimiter=' ')
            if(options.verbose): print 'reading %s' % (iwmFileName)
        except:
            print 'unable to read %s' % (iwmFileName)
            incompleteData = True
            return
        #stack the category data to the existing data-set data
        if(iwmData==None):
            iwmData = iwmCatData
        else:
            iwmData = np.concatenate((iwmData,iwmCatData),axis=0)
        pass
    
    if(incompleteData==False):
        if(options.verbose): print 'co-clustering...'
        ccData = coclust(iwmData, dataset, nRowCluster, nTopic, ccType)
        # the indices of co-clusters columns
        ccarray = ccData[1].split()
        ccCol = np.array(ccarray,dtype=np.int16)
        if(options.verbose): print 'writing %s'% (resultFileName)
        np.savetxt(resultFileName, ccCol, fmt = '%d', delimiter=' ')
    else:
        print 'incomplete data for %s' % (resultFileName)

if __name__ == '__main__':
    for iWord,wordn in enumerate(nWords):
        for iTopic,topicn in enumerate(nTopics):
            ccUniversalTopicDictionary(wordn,topicn)
pass