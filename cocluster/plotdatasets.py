'''
Created on 18 Oct 2011

@author: ag00087
'''

import shelve
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#global variables
rootDir = '/vol/vssp/diplecs/ash/Data/'
dataDir = '/results/icpram/'
confDir = 'icpram2012/'
datasets = ['VOC2006','VOC2007','VOC2010','Scene15','Caltech101','Caltech256']
figfmt = 'png'
# the data in shelved

def plotperfvsdataset(ccType,nCodeword):
    ndatasets = len(datasets)
    ccMeans = np.zeros(ndatasets)
    ccStds = np.zeros(ndatasets)
    blMeans = np.zeros(ndatasets)
    blStds = np.zeros(ndatasets)
    
    
    for i,dataset in enumerate(datasets):
        resultfile = '{0}{1}{2}{3}{4}{5}'.format(rootDir,dataset,dataDir,'ccknnWord',ccType,str(nCodeword))
        try:
            result = shelve.open(resultfile)
            try:
                ccMean = result['ccMean']
                ccStd = result['ccStd']
                blMean = result['blMean']
                blStd = result['blStd']
            except:
                print 'unable to access shelved item %s : %d : %s' % (ccType,nCodeword,dataset)
            result.close()
        except:
            print 'unable to open shelved file %s : %d' % (ccType,nCodeword)
        ccMeans[i] = ccMean
        ccStds[i] = ccStd
        blMeans[i] = blMean
        blStds[i] = blStd
    
    result1 = [ccMeans,ccStds]
    result2 = [blMeans,blStds]
    
    figTitle = '{0}{1}{2}'.format('ccblWord',ccType,str(nCodeword))
    outPath = '{0}{1}{2}{3}{4}{5}{6}{7}'.format(rootDir,confDir,'ccWord',ccType,str(nCodeword),'BAR','.',figfmt)
    try:
        #plotWordResult(result1,result2,datasets,outPath,figTitle,figfmt,ccType,nCodeword)
        plotWordResultBAR(result1,result2,datasets,outPath,figTitle,figfmt,ccType,nCodeword)
    except:
        print 'unable to plot result, please experiment again'
        
    #collate data for the 3d bar plot
    
    pass

def plotWordResultBAR(result1,result2,xticklabels,outPath,title,figfmt,ccType,nCodeword):
    nXTicks = len(xticklabels)
    fig = plt.figure()
    figfmt = 'png'
    ax = fig.add_subplot(111)
    ind = np.arange(nXTicks)  # the x locations for the groups
    width = 0.35       # the width of the bars
    ax.set_xticks(np.arange(nXTicks))
    ax.set_xticklabels(xticklabels,rotation=0,size='medium',ha='center')
    cclabel = '%s:%s: %d' % ('CC',ccType,nCodeword)
    boflabel = 'BoW: %d' % (nCodeword)
    rects1 = ax.bar(ind, result2[0], width, color='y',label=boflabel)
    rects2 = ax.bar(ind+width, result1[0], width, color='k',label=cclabel)
    plt.xlabel('Data Set',size='large')
    plt.ylabel('F1_score',size='large')
#    plt.title('%s ' % dataset,size='large')
    plt.legend(loc="upper right")
    maxf1 = np.max([result1[0],result2[0]])+0.1*np.mean([result1[0],result2[0]])
    minf1 = np.min([result1[0],result2[0]])-0.1*np.mean([result1[0],result2[0]])
    plt.ylim([minf1,maxf1])
    ax.legend( (rects1[0], rects2[0]), (boflabel, cclabel) )
    ax.set_xticks(ind+width)
    plt.savefig(outPath,format=figfmt)
    plt.show()
    plt.close()
    pass

def plotWordResult(result1,result2,xticklabels,outPath,title,figfmt,ccType,nCodeword):
    nXTicks = len(xticklabels)
    ax = plt.subplot(111)
    #plt.errorbar(np.arange(1,(nXTicks+1)), result1[0], result1[1][0], fmt = '-', color= 'r', ecolor='r', elinewidth=1, marker = 'x',markerfacecolor='k', label = 'CoCluster-SVM')
    #plt.errorbar(np.arange(1,(nXTicks+1)), result2[0][0], result2[1][0], fmt = '-', color= 'b', ecolor='b', elinewidth=1, marker = 'o',markerfacecolor='k', label = 'BoF-SVM')
    
    cclabel = 'co-cluster: %s : %d' % (ccType,nCodeword)
    boflabel = 'BoF: %d' % (nCodeword)
    plt.plot(np.arange(2,(nXTicks+2)), result1[0],'k-o',linewidth=2.5, label = cclabel, markersize = 8)
    plt.plot(np.arange(2,(nXTicks+2)), result2[0],'k--d',linewidth=2.5, label = boflabel, markersize = 8)
    plt.xlabel('Data set', size='large')
    plt.ylabel('F1_score', size='large')
    
    plt.legend(loc="upper right")
    plt.ylim([0.2,0.4])
    ax.set_xticks(np.arange(2,(nXTicks+2)))
    ax.set_xticklabels(xticklabels,size='medium',ha='center')
    plt.savefig(outPath,format=figfmt)
    plt.show()
    print 'plotted'
    plt.close()
    
def plotSchemevsNumwords():
    ccTypes = ['i','e','r']
    nCodewords = [100,500,1000]
    ccBar = np.zeros((len(ccTypes),len(nCodewords)))
        
    for i,ccType in enumerate(ccTypes):
        for j,nCodeword in enumerate(nCodewords):
            ccBar[i,j] = findMeanF1overDatasets(ccType,nCodeword)
    
    coclusterBarPlot(ccBar,ccTypes,nCodewords)
    
    
def findMeanF1overDatasets(ccType,nCodeword):
    ccMeans = np.zeros(6)
    ccStds = np.zeros(6)
    blMeans = np.zeros(6)
    blStds = np.zeros(6)
    
    for i,dataset in enumerate(datasets):
        resultfile = '{0}{1}{2}{3}{4}{5}'.format(rootDir,dataset,dataDir,'ccknnWord',ccType,str(nCodeword))
        try:
            result = shelve.open(resultfile)
            try:
                ccMean = result['ccMean']
                ccStd = result['ccStd']
                blMean = result['blMean']
                blStd = result['blStd']
            except:
                print 'unable to access shelved item %s : %d : %s' % (ccType,nCodeword,dataset)
            result.close()
        except:
            print 'unable to open shelved file %s : %d' % (ccType,nCodeword)
        ccMeans[i] = ccMean
        ccStds[i] = ccStd
        blMeans[i] = blMean
        blStds[i] = blStd
        
    return np.mean(ccMeans)
    
def coclusterBarPlot(ccBar,ccTypes,nCodewords):
#    ccTypes = ['i','e','r']
#    nCodewords = [100,500,1000]
    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
    ax = Axes3D(fig)
    i = 0
    for c, z in zip(['r','g','b'], nCodewords):
        xs = np.arange(ccBar.shape[0])
        ys = ccBar[i,:]
        i = i+1
        print ys
        # You can provide either a single color or an array. To demonstrate this,
        # the first bar of each set will be colored cyan.
        cs = [c] * len(xs)
        cs[0] = 'c'
        ax.bar(xs, ys, zs=z, zdir='y', color=cs, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()

    pass

def plotperfvsdatasetmain():
    ccTypes = ['e','i','r']
    nCodewords = [100,500,1000,5000]
    
    for ccType in ccTypes:
        for nCodeword in nCodewords:
            plotperfvsdataset(ccType,nCodeword)
    
if __name__ == '__main__':
    plotperfvsdatasetmain()
    #plotSchemevsNumwords()
    pass
            