'''
Created on 22 Feb 2012
Plot results for visapp using bar graph
@author: ag00087
'''
import shelve
import matplotlib.pyplot as plt
import numpy as np
#global variables
rootDir = '/vol/vssp/diplecs/ash/Data/'
#dataDir = '/results/'
dataDir = '/results/visapp/'
#confDir = 'visapp2012/perfDataset/'
#confDir = 'visapp2012/perfDictSize/'
confDir = 'visapp2012/barplot/'
#datasets = ['VOC2006','VOC2007','VOC2010','Scene15','Caltech101','Caltech256']
datasets = ['VOC2006','VOC2007','VOC2010','Scene15']

figfmt = 'png'
ccTypes = ['e','i','r']
nWord = 10000
nCodewords = [100,500,1000,5000]
#nCodewords = [50,100,500,1000,5000]
#nCodewords = [50]

# the combinatorial parameters are datasets,ccTypes,nCodewords
# the total number of graphs is 6 * 3 * 4 = 72

def plotF1Dataset(dataset,ccType,nCodeword):
    resultfile = '{0}{1}{2}{3}{4}{5}{6}'.format(rootDir,dataset,dataDir,'ccknnTopic',ccType,str(nCodeword),str(nWord))
    try:
        result = shelve.open(resultfile)
        try:
            catList = result['catList']
            ccResultMean = result['ccResultMean'] 
            ccResultStd = result['ccResultStd']
            blResultMean = result['blResultMean']
            blResultStd = result['blResultStd']
            
        except:
            print 'unable to access shelved item %s : %d : %s' % (ccType,nCodeword,dataset)
        result.close()
    except:
        print 'unable to open shelved file %s : %d' % (ccType,nCodeword)
    outPath = '{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(rootDir,confDir,'ccknnTopic',dataset,ccType,str(nCodeword),'BAR','.',figfmt)
    #plotF1DatasetResult([ccResultMean,ccResultStd],[blResultMean,blResultStd],catList,dataset,outPath,figfmt,ccType,nCodeword)
    barPlotF1DatasetResult([ccResultMean,ccResultStd],[blResultMean,blResultStd],catList,dataset,outPath,figfmt,ccType,nCodeword)
    pass

def plotF1DatasetResult(result1,result2,xticklabels,dataset,outPath,figfmt,ccType,nCodeword):
    nXTicks = len(xticklabels)
    ax = plt.subplot(111)
    cclabel = 'Bag-of-Topics, %d' % (nCodeword)
    boflabel = 'Bag-of-Words, %d' % (nCodeword)
    plt.plot(np.arange(1,(nXTicks+1)), result1[0],'r-o',linewidth=2.5, label = cclabel)
    plt.plot(np.arange(1,(nXTicks+1)), result2[0],'k--*',linewidth=2.5, label = boflabel)
    plt.xlabel('visual category',size='large')
    plt.ylabel('F1_score',size='large')
#    plt.title('%s Performance: %s ' % (title,dataset))
    plt.title('%s ' % dataset,size='large')
    plt.legend(loc="upper right")
    plt.ylim([0.1,0.6])
    ax.set_xticks(np.arange(0,(nXTicks+1)))
    ax.set_xticklabels(xticklabels,rotation=20,size='large',ha='center')
    plt.savefig(outPath,format=figfmt)
    plt.show()
    plt.close()
    
def barPlotF1DatasetResult(result1,result2,xticklabels,dataset,outPath,figfmt,ccType,nCodeword):
    maxacc = np.max([result1,result2])+0.05
    minacc = np.min([result1,result2])-0.05
    nXTicks = len(xticklabels)
    cclabel = '%s:%s: %d' % ('CC',ccType,nCodeword)
    boflabel = 'BoW: %d' % (nCodeword)
           
    ind = np.arange(0,nXTicks)  # the x locations for the groups
    width = 0.35       # the width of the bars
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
#    fig.autoscale(enable=True, axis='both', tight=False)
    ax.autoscale_view(True,True,True)

    rects1 = ax.bar(ind, result2[0], width, color='y',label=boflabel)
    rects2 = ax.bar(ind+width, result1[0], width, color='k',label=cclabel)
    ax.legend( (rects2[0], rects1[0]), ('BoW','SSRBC') )

    ax.set_xticks(ind+width)
    plt.xlabel('visual category',size='large')
    plt.ylabel('F1_score',size='large')
    plt.title('%s ' % dataset,size='large')
    plt.legend(loc="upper right")
    
    plt.ylim([0.1,maxacc])
#    plt.ylim([0.1,0.6])
    
    ax.set_xticklabels(xticklabels,rotation=25,size='large',ha='right')
    
    plt.savefig(outPath,format='png')
    
    plt.show()
    
def plotperfvisapp():
    for dataset in (datasets):
        for ccType in (ccTypes):
            for nCodeword in (nCodewords):
                try:
                    plotF1Dataset(dataset,ccType,nCodeword)
                    print 'plotted: %s : %s : %d' % (dataset,ccType,nCodeword)
                except:
                    print 'unable to plot for %s : %s : %d' %(dataset,ccType,nCodeword)

if __name__ == '__main__':
    plotperfvisapp()
    pass