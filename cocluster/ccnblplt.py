'''
Created on 2 Aug 2011
plot error bar of mean average precision performance  
@author: ag00087
'''

import matplotlib.pyplot as plt
import numpy as np

rootDir = '/vol/vssp/diplecs/ash/Data/'
outDir = '/results/'

def plotresult(dataset,xticklabels,result1,result2,figfmt='svg',title='CoClusterBagofFeatures',ccType='i'):
    nXTicks = len(xticklabels)
    outPath = rootDir + dataset + outDir + '%s%s%s%s%s'%(title,dataset,ccType,'.',figfmt) 
    plt.figure()
    ax = plt.subplot(111)
    plt.errorbar(np.arange(1,(nXTicks+1)), result1[0], result1[1], fmt = '-', color= 'r', ecolor='r', elinewidth=1, marker = 'x',markerfacecolor='k', label = 'CoCluster')
    plt.errorbar(np.arange(1,(nXTicks+1)), result2[0], result2[1], fmt = '-', color= 'b', ecolor='b', elinewidth=1, marker = 'o',markerfacecolor='k', label = 'BoF')
    plt.xlabel('Visual Categories')
    plt.ylabel('Performance Metric')
    plt.title('%s Performance: %s ' % (title,dataset))
    plt.legend(loc="lower right")
    plt.ylim([0.0,1.0])
    ax.set_xticks(np.arange(1,(nXTicks+2)))
    ax.set_xticklabels(xticklabels,rotation=30,size='small',ha='center')
    plt.savefig(outPath,format=figfmt)
    plt.show()
    plt.close()
    
def plotClassifierResult(result1,result2,xticklabels,dataset,outPath,title,figfmt):
    nXTicks = len(xticklabels)
    ax = plt.subplot(111)
    #plt.errorbar(np.arange(1,(nXTicks+1)), result1[0], result1[1][0], fmt = '-', color= 'r', ecolor='r', elinewidth=1, marker = 'x',markerfacecolor='k', label = 'CoCluster-SVM')
    #plt.errorbar(np.arange(1,(nXTicks+1)), result2[0][0], result2[1][0], fmt = '-', color= 'b', ecolor='b', elinewidth=1, marker = 'o',markerfacecolor='k', label = 'BoF-SVM')
    plt.errorbar(np.arange(1,(nXTicks+1)), result1[0], result1[1], fmt = '-', color= 'r', ecolor='r', elinewidth=1, marker = 'x',markerfacecolor='k', label = 'CoCluster-KNN')
    plt.errorbar(np.arange(1,(nXTicks+1)), result2[0], result2[1], fmt = '-', color= 'b', ecolor='b', elinewidth=1, marker = 'o',markerfacecolor='k', label = 'BoF-KNN')
    plt.xlabel('Visual Categories')
    plt.ylabel('fBeta_score')
    plt.title('%s Performance: %s ' % (title,dataset))
    plt.legend(loc="lower right")
    plt.ylim([0.0,1.0])
    ax.set_xticks(np.arange(1,(nXTicks+2)))
    ax.set_xticklabels(xticklabels,rotation=30,size='small',ha='center')
    plt.savefig(outPath,format=figfmt)
    plt.show()
    plt.close()
    
def plotWordResult(result1,result2,xticklabels,dataset,outPath,title,figfmt):
    nXTicks = len(xticklabels)
    ax = plt.subplot(111)
    #plt.errorbar(np.arange(1,(nXTicks+1)), result1[0], result1[1][0], fmt = '-', color= 'r', ecolor='r', elinewidth=1, marker = 'x',markerfacecolor='k', label = 'CoCluster-SVM')
    #plt.errorbar(np.arange(1,(nXTicks+1)), result2[0][0], result2[1][0], fmt = '-', color= 'b', ecolor='b', elinewidth=1, marker = 'o',markerfacecolor='k', label = 'BoF-SVM')
    plt.errorbar(np.arange(1,(nXTicks+1)), result1[0], result1[1], fmt = '-', color= 'r', ecolor='r', elinewidth=1, marker = 'x',markerfacecolor='k', label = 'Co-Cluster')
    plt.errorbar(np.arange(1,(nXTicks+1)), result2[0], result2[1], fmt = '-', color= 'b', ecolor='b', elinewidth=1, marker = '+',markerfacecolor='k', label = 'Base-line')
    plt.xlabel('visual categories')
    plt.ylabel('F1_score')
#    plt.title('%s Performance: %s ' % (title,dataset))
    plt.title('%s ' % dataset)
    plt.legend(loc="upper right")
    plt.ylim([0.0,1.0])
    ax.set_xticks(np.arange(1,(nXTicks+2)))
    ax.set_xticklabels(xticklabels,rotation=30,size='small',ha='center')
    plt.savefig(outPath,format=figfmt)
    plt.show()
    plt.close()
    
def plotTopicResult(result1,result2,xticklabels,dataset,outPath,figfmt):
    nXTicks = len(xticklabels)
    ax = plt.subplot(111)
    plt.errorbar(np.arange(1,(nXTicks+1)), result1[0], result1[1], fmt = '-', color= 'r', ecolor='r', elinewidth=1, marker = 'x',markerfacecolor='k', label = 'cocluster-Topic')
    plt.errorbar(np.arange(1,(nXTicks+1)), result2[0], result2[1], fmt = '-', color= 'b', ecolor='b', elinewidth=1, marker = '+',markerfacecolor='k', label = 'Bag-of-Words')
    plt.xlabel('visual categories')
    plt.ylabel('F1_score')
    plt.title('%s ' % dataset)
    plt.legend(loc="upper right")
    plt.ylim([0.0,1.0])
    ax.set_xticks(np.arange(1,(nXTicks+2)))
    ax.set_xticklabels(xticklabels,rotation=30,size='small',ha='center')
    plt.savefig(outPath,format=figfmt)
    plt.show()
    plt.close()
    
def plotKNNresult(result1,result2,xticklabels,dataset,outPath,title,figfmt):
    nXTicks = len(xticklabels)
    ax = plt.subplot(111)
    #plt.errorbar(np.arange(1,(nXTicks+1)), result1[0], result1[1][0], fmt = '-', color= 'r', ecolor='r', elinewidth=1, marker = 'x',markerfacecolor='k', label = 'CoCluster-SVM')
    #plt.errorbar(np.arange(1,(nXTicks+1)), result2[0][0], result2[1][0], fmt = '-', color= 'b', ecolor='b', elinewidth=1, marker = 'o',markerfacecolor='k', label = 'BoF-SVM')
    plt.plot(np.arange(1,(nXTicks+1)), result1, fmt = '-', color= 'r', marker = 'x',markerfacecolor='k', label = 'CoCluster-KNN')
    plt.plot(np.arange(1,(nXTicks+1)), result2, fmt = '-', color= 'b', marker = 'o',markerfacecolor='k', label = 'BoF-KNN')
    plt.xlabel('Visual Categories')
    plt.ylabel('F1_score')
    plt.title('%s Performance: %s ' % (title,dataset))
    plt.legend(loc="upper right")
    plt.ylim([0.0,1.0])
    ax.set_xticks(np.arange(1,(nXTicks+2)))
    ax.set_xticklabels(xticklabels,rotation=30,size='small',ha='center')
    plt.savefig(outPath,format=figfmt)
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    pass