'''
Created on 15 Nov 2011

@author: ag00087
'''
'''
Created on 15 Nov 2011

@author: ag00087
'''

# import libraries
import numpy as np
from optparse import OptionParser
import sys
from ash.cocluster.cocluster import coclust

# parse options
parser = OptionParser()
parser.add_option('-d','--dataset',action='store',type='string',dest='dataset',metavar='dataset',help='visual dataset')
parser.add_option('-w','--nCodeword',action='store',type='int',dest='nCodeword',default=1024,metavar='nCodeword',help='number of code words: 1024,2048,4096,8192,16384')
parser.add_option('-t','--nTopic',action='store',type='int',dest='nTopic',metavar='nTopic',default=128,help='number of topic words')
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
ucbDir = '/UniversalCB/'
utdDir = '/UniversalTopicDictionary/'


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
    kernelType = options.kernelType
    nFold = options.nFold
    nCodeword = options.nCodeword
    beta = options.beta
    
    #echo arguments    
    if(options.verbose): 
        print dataset,nRowCluster,nTopic,ccType,kernelType,beta,nFold,nCodeword
        print options
        
    #configure data path and other parameters
    dataPath = rootDir+dataset+imgWrdDir
    resultPath = rootDir+dataset+utdDir+dataset
    catmap = getCatMap(dataset)
    catList = catmap.keys()
    dataext = str(nCodeword)+imgWrdext
    resultext = str(nCodeword)+utdext
    
    
    # initialise empty iwm matrix and append each category to it
    iwmData = np.empty()
    for catName in catList:
        iwmFileName = dataPath+catName+dataext
        iwmCatData = np.loadtxt(iwmFileName,dtype=np.int16,delimiter=' ')
        #stack the category data to the existing data-set data
        iwmData = np.concatenate((iwmData,iwmCatData),axis=0)
        pass
    
    if(options.verbose): print 'co-clustering...'
            
    ccData = coclust(iwmData, dataset, nRowCluster, nTopic, ccType)
    # the indices of co-clusters columns
    ccCol = np.array([int(i) for i in ccData[1].split()])
    
    resultFileName = resultPath+resultext
    np.savetxt(resultFileName, ccCol, fmt=np.int, delimiter=' ')

if __name__ == '__main__':
    ccUniversalTopicDictionary()
    pass
        