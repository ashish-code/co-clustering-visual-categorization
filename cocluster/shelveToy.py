'''
Created on 18 Aug 2011

@author: ag00087
'''
import shelve

datapath = '/vol/vssp/diplecs/ash/Data/Caltech256/results/ccknnWordr5000'
db = shelve.open(datapath)
#catList = db['catList']
ccResultMean = db['ccResultMean']
ccResultStd = db['ccResultStd']
blResultMean = db['blResultMean']
blResultStd = db['blResultStd']

db.close()
#print catList
print ccResultMean
print blResultMean


if __name__ == '__main__':
    pass