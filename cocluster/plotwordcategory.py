'''
Created on 21 Oct 2011

@author: ag00087
'''
import shelve
import matplotlib.pyplot as plt
import numpy as np
#global variables
rootDir = '/vol/vssp/diplecs/ash/Data/'
dataDir = '/results/icpram/'
confDir = 'icpram2012/perfDataset/'
datasets = ['VOC2006','VOC2007','VOC2010','Scene15','Caltech101','Caltech256']
figfmt = 'png'
ccTypes = ['i','e','r']
nCodewords = [100,500,1000,5000]

# the combinatorial parameters are datasets,ccTypes,nCodewords
# the total number of graphs is 6 * 3 * 4 = 72

def plotF1Dataset(dataset,ccType,nCodeword):
    resultfile = '{0}{1}{2}{3}{4}{5}'.format(rootDir,dataset,dataDir,'ccknnWord',ccType,str(nCodeword))
    try:
        result = shelve.open(resultfile)
        try:
            catList = result['catList']
            ccResultMean = result['ccResultMean'] 
            ccResultStd = result['ccResultStd']
            blResultMean = result['blResultMean']
            blResultStd = result['blResultStd']
            #result['ccMean'] = ccMean
            #result['ccStd'] = ccStd
            #result['blMean'] = blMean
            #result['blStd'] = blStd
        except:
            print 'unable to access shelved item %s : %d : %s' % (ccType,nCodeword,dataset)
        result.close()
    except:
        print 'unable to open shelved file %s : %d' % (ccType,nCodeword)
    outPath = '{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(rootDir,confDir,'ccknnWord',dataset,ccType,str(nCodeword),'BAR','.',figfmt)
    plotF1DatasetResultBAR([ccResultMean,ccResultStd],[blResultMean,blResultStd],catList,dataset,outPath,figfmt,ccType,nCodeword)
    pass

def plotF1DatasetResult(result1,result2,xticklabels,dataset,outPath,figfmt,ccType,nCodeword):
    nXTicks = len(xticklabels)
    ax = plt.subplot(111)
    
    cclabel = 'CC: %s : %d' % (ccType,nCodeword)
    boflabel = 'BoF: %d' % (nCodeword)
    plt.plot(np.arange(1,(nXTicks+1)), result1[0],'k-o',linewidth=2.0, label = cclabel, markersize = 8)
    plt.plot(np.arange(1,(nXTicks+1)), result2[0],'k--d',linewidth=2.0, label = boflabel, markersize = 8)
    
    plt.xlabel('visual categories',size='large')
    plt.ylabel('F1_score',size='large')
#    plt.title('%s Performance: %s ' % (title,dataset))
    plt.title('%s ' % dataset,size='large')
    plt.legend(loc="upper right")
    plt.ylim([0.1,0.6])
    ax.set_xticks(np.arange(1,(nXTicks+2)))
    ax.set_xticklabels(xticklabels,rotation=30,size='medium',ha='center')
    plt.savefig(outPath,format=figfmt)
    plt.show()
    plt.close()
    
def plotF1DatasetResultBAR(result1,result2,xticklabels,dataset,outPath,figfmt,ccType,nCodeword):
    nXTicks = len(xticklabels)
    fig = plt.figure()
    figfmt = 'png'
    ax = fig.add_subplot(111)
    ind = np.arange(nXTicks)  # the x locations for the groups
    width = 0.35       # the width of the bars
    ax.set_xticks(np.arange(nXTicks))
    ax.set_xticklabels(xticklabels,rotation=20,size='medium',ha='center')
    cclabel = 'SSRBC: %d' % (nCodeword)
    boflabel = 'BoW: %d' % (nCodeword)
    rects1 = ax.bar(ind, result2[0], width, color='y',label=boflabel)
    rects2 = ax.bar(ind+width, result1[0], width, color='k',label=cclabel)
    plt.xlabel('visual categories',size='large')
    plt.ylabel('F1_score',size='large')
    plt.title('%s ' % dataset,size='large')
    plt.legend(loc="upper right")
    maxf1 = np.max([result1[0],result2[0]])+0.1*np.mean([result1[0],result2[0]])
    minf1 = np.min([result1[0],result2[0]])-0.1*np.mean([result1[0],result2[0]])
    plt.ylim([minf1,maxf1])
    #plt.ylim([0.1,0.6])
    ax.legend( (rects1[0], rects2[0]), (boflabel, cclabel) )
    ax.set_xticks(ind+width)
    #ax.set_xticks(np.arange(1,(nXTicks+2)))
    #ax.set_xticklabels(xticklabels,rotation=30,size='medium',ha='center')
    plt.savefig(outPath,format=figfmt)
    plt.show()
    plt.close()
    pass
    
def plotperficpram():
    for dataset in (datasets):
        for ccType in (ccTypes):
            for nCodeword in (nCodewords):
                try:
                    plotF1Dataset(dataset,ccType,nCodeword)
                    print 'plotted: %s : %s : %d' % (dataset,ccType,nCodeword)
                except:
                    print 'unable to plot for %s : %s : %d' %(dataset,ccType,nCodeword)

if __name__ == '__main__':
    plotperficpram()
    pass