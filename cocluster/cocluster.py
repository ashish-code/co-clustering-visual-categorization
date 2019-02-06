'''
Created on 27 Jul 2011

@author: ag00087
'''

import os
import numpy as np
import time

rootDir = '/vol/vssp/diplecs/ash/Data/'
tempDir = 'Temp'

def Oldcoclust(data,dataset,nRowCluster=1,nColCluster=1,ccType='i'):
    
    tempPath = rootDir+dataset+tempDir
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
    return tempFileData   

def coclust(data,dataset,nRowCluster=1,nColCluster=1,ccType='i'):
    
    tempTimeDir = str(int(time.time()))
    tempPath = rootDir+tempDir+'/'+tempTimeDir+'/'
    tempdataPath = tempPath+'tempdata'
    os.mkdir(tempPath)
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
    return tempFileData   
    
def coclustWord(data,dataset,nRowCluster=1,nColCluster=1,ccType='i'):
    
    tempPath = rootDir+dataset+tempDir
    tempdataPath = tempPath+'tempdata'+str(ccType)+str(nRowCluster)+str(nColCluster)
    tempdataDimPath = tempPath+'tempdata'+str(ccType)+str(nRowCluster)+str(nColCluster)+'_dim'
    tempCCFilePath = tempPath+'tempCCFile'+str(ccType)+str(nRowCluster)+str(nColCluster)
    np.savetxt(tempdataPath,data,fmt='%d',delimiter=' ')
    np.savetxt(tempdataDimPath,data.shape,fmt='%d',delimiter=' ')
    
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
    return tempFileData   


if __name__ == '__main__':
    pass