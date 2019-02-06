import shelve
import matplotlib.pyplot as plt
import numpy as np
#global variables
rootDir = '/vol/vssp/diplecs/ash/Data/'
#dataDir = '/results/'
dataDir = '/results/visapp/'
#confDir = 'visapp2012/perfDataset/'
confDir = 'visapp2012/perfDictSize/'
datasets = ['VOC2006','VOC2007','VOC2010','Scene15','Caltech101','Caltech256']
#datasets = ['VOC2006','VOC2007','VOC2010','Scene15']
#datasets = ['Scene15']
#datasets = ['VOC2007']
figfmt = 'png'
ccTypes = ['r']
nWord = 10000
nCodewords = [100,500,1000,5000]
#nCodewords = [50,100,500,1000,5000]
#nCodewords = [500]

# the combinatorial parameters are datasets,ccTypes,nCodewords
# the total number of graphs is 6 * 3 * 4 = 72

def plotF1Dataset(dataset,ccType,nCodeword):
    #resultfile = '{0}{1}{2}{3}{4}{5}{6}'.format(rootDir,dataset,dataDir,'ccblTopic',ccType,str(nCodeword),str(nWord))
    resultfile = '{0}{1}{2}{3}{4}{5}{6}'.format(rootDir,dataset,dataDir,'ccknnTopic',ccType,str(nCodeword),str(nWord))
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
    #outPath = '{0}{1}{2}{3}{4}{5}{6}{7}'.format(rootDir,confDir,'ccblTopic',dataset,ccType,str(nCodeword),'.',figfmt)
    outPath = '{0}{1}{2}{3}{4}{5}{6}{7}{8}'.format(rootDir,confDir,'ccknnTopic',dataset,ccType,str(nCodeword),'BAR','.',figfmt)
    #plotF1DatasetResult([ccResultMean,ccResultStd],[blResultMean,blResultStd],catList,dataset,outPath,figfmt,ccType,nCodeword)
    #plotF1DatasetResultBAR([ccResultMean,ccResultStd],[blResultMean,blResultStd],catList,dataset,outPath,figfmt,ccType,nCodeword)
    pass

def plotF1DatasetResult(result1,result2,xticklabels,dataset,outPath,figfmt,ccType,nCodeword):
    nXTicks = len(xticklabels)
    ax = plt.subplot(111)
    #plt.errorbar(np.arange(1,(nXTicks+1)), result1[0], result1[1][0], fmt = '-', color= 'r', ecolor='r', elinewidth=1, marker = 'x',markerfacecolor='k', label = 'CoCluster-SVM')
    #plt.errorbar(np.arange(1,(nXTicks+1)), result2[0][0], result2[1][0], fmt = '-', color= 'b', ecolor='b', elinewidth=1, marker = 'o',markerfacecolor='k', label = 'BoF-SVM')
    cclabel = 'Bag-of-Topics, %d' % (nCodeword)
    boflabel = 'Bag-of-Words, %d' % (nCodeword)
    plt.plot(np.arange(1,(nXTicks+1)), result1[0],'r-o',linewidth=2.5, label = cclabel)
    plt.plot(np.arange(1,(nXTicks+1)), result2[0],'k--*',linewidth=2.5, label = boflabel)
    #plt.errorbar(np.arange(1,(nXTicks+1)), result1[0], result1[1], fmt = '-', linewidth=2.0,color= 'r', ecolor='r', elinewidth=1, marker = 'o',markerfacecolor='r', label = cclabel)
    #plt.errorbar(np.arange(1,(nXTicks+1)), result2[0], result2[1], fmt = '--', linewidth=2.0,color= 'b', ecolor='b', elinewidth=1, marker = 'd',markerfacecolor='b', label = boflabel)
    plt.xlabel('visual category',size='large')
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
    
def plotTopicResultBAR(result1,result2,xticklabels,outPath,title,figfmt,ccType,nCodeword):
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
    
def plotperfvisapp():
    for dataset in (datasets):
        for ccType in (ccTypes):
            for nCodeword in (nCodewords):
                try:
                    plotF1Dataset(dataset,ccType,nCodeword)
                    print 'plotted: %s : %s : %d' % (dataset,ccType,nCodeword)
                except:
                    print 'unable to plot for %s : %s : %d' %(dataset,ccType,nCodeword)


def plotperfvsdatasetmain():
    ccTypes = ['e','i','r']
    nCodewords = [100,500,1000,5000]
    
    for ccType in ccTypes:
        for nCodeword in nCodewords:
            plotperfvsdataset(ccType,nCodeword)


def plotperfvsdataset(ccType,nCodeword):
    ndatasets = len(datasets)
    ccMeans = np.zeros(ndatasets)
    ccStds = np.zeros(ndatasets)
    blMeans = np.zeros(ndatasets)
    blStds = np.zeros(ndatasets)
    
    
    for i,dataset in enumerate(datasets):
        resultfile = '{0}{1}{2}{3}{4}{5}{6}'.format(rootDir,dataset,dataDir,'ccknnTopic',ccType,str(nCodeword),str(nWord))
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
    
    figTitle = '{0}{1}{2}'.format('ccbllTopic',ccType,str(nCodeword))
    outPath = '{0}{1}{2}{3}{4}{5}{6}{7}'.format(rootDir,confDir,'ccTopic',ccType,str(nCodeword),'BAR','.',figfmt)
    try:
        #plotWordResult(result1,result2,datasets,outPath,figTitle,figfmt,ccType,nCodeword)
        plotTopicResultBAR(result1,result2,datasets,outPath,figTitle,figfmt,ccType,nCodeword)
    except:
        print 'unable to plot result, please experiment again'
        
    #collate data for the 3d bar plot
    
    pass
    
if __name__ == '__main__':
#    plotperfvisapp()
    plotperfvsdatasetmain()
    pass