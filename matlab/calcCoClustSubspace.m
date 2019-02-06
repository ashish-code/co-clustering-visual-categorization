% compute bregman co-clustering for sampled data and learnt coefficients
% The results of co-clustering will be utilized in induction of structured
% sparsity during structured sparse decomposition
function calcCoClustSubspace(dataSet,dictType,dictSize,sampleSize,rowClust,colClust,ccType)
% function calcCoClustSubspace(dataSet,dictType,dictSize,sampleSize)
% dataSet: VOC2006,VOC2007,VOC2010,Scene15,Caltech101,Caltech256
% dictType: universal,categorical,balanced
% dictSize: 1000
% sampleSize: 100000
%---------------------------------------------------------------------
% initialize matlab
cdir = pwd;
cd ~;
startup;
cd (cdir);
%---------------------------------------------------------------------
% paths to data directories
paths.rootDir = '/vol/vssp/diplecs/ash/Data/';
paths.sampleDir = '/collated/';
paths.dictDir = '/Dictionary/';
paths.coclustDir = '/CoClust/';
paths.coeffDir = '/Coeff/';
paths.imageListDir = '/ImageLists/';
paths.tempDir = 'Temp/';
%---------------------------------------------------------------------
params.dataSet = dataSet;
params.dictType = dictType;
params.dictSize = dictSize;
params.sampleSize = sampleSize;
params.categoryListFileName = 'categoryList.txt';
params.rowClust = rowClust;
params.colClust = colClust;
params.ccType = ccType;
params.progPath = '/vol/vssp/diplecs/ash/code/cocluster/';
params.prog = 'cocluster-linux';
%---------------------------------------------------------------------
% the pipeline depends extensively upon the dictionary type
if strcmp(dictType,'universal')
    callCoClustSubspaceUniversal(params,paths);
elseif strcmp(dictType,'categorical')
    callCoClustSubspaceCategorical(params,paths);
elseif strcmp(dictType,'balanced')
    callCoClustSubspaceBalanced(params,paths);
end
%---------------------------------------------------------------------

end

function callCoClustSubspaceUniversal(params,paths)
% read the sampled file and run co-clustering accoringly
sampleDataFile = [(paths.rootDir),(params.dataSet),(paths.sampleDir),(params.dataSet),num2str(params.sampleSize),'.uni'];
sampleData = load(sampleDataFile);
fprintf('%s loaded\n',sampleDataFile);

nVec = size(sampleData,2);
nSample = 10000;
rndSample = randsample(nVec,nSample);
sampleData = sampleData(:,rndSample);

tempTimeDir = strcat(num2str(floor(now*10000000)),'/');
tempPath = strcat(paths.rootDir,paths.tempDir,tempTimeDir);
if ~exist(tempPath,'dir')
    mkdir(tempPath);
end
tempDataPath = strcat(tempPath,'tempdata');
if ~exist(tempDataPath,'file')
    dlmwrite(tempDataPath,sampleData,' ');
end
tempDataDimPath = strcat(tempPath,'tempdata_dim');
dataDim = size(sampleData)';
if ~exist(tempDataDimPath,'file')
    fid = fopen(tempDataDimPath,'w');
    fprintf(fid,'%s\n%s',num2str(dataDim(1)),num2str(dataDim(2)));
    fclose(fid);
end
tempCCFilePath = strcat(paths.rootDir,params.dataSet,paths.coclustDir,params.dataSet,num2str(params.dictSize),params.dictType,num2str(params.sampleSize),params.ccType,num2str(params.rowClust),num2str(params.colClust),'.s');

if exist(tempCCFilePath,'file')
    return;
end

progArgs = sprintf(' -A %s -R %d -C %d -I d s %s -O c s 0 o %s',params.ccType,params.rowClust,params.colClust,tempDataPath,tempCCFilePath);
cmd = strcat(params.progPath,params.prog,progArgs);
% system call to cocluster linux program
fprintf('%s\n','co-clustering...');
system(cmd);
if exist(tempCCFilePath,'file')
    fprintf('%s written\n',tempCCFilePath);
    rmdir(tempPath,'s');    
else
    fprintf('%s ERROR\n',tempCCFilePath);
end

end

function callCoClustSubspaceCategorical(params,paths)
% for each category in the dataset, read the sampled file and run
% co-clustering accoringly
% read the category list in the dataset
categoryListPath = [(paths.rootDir),(params.dataSet),'/',(params.categoryListFileName)];
fid = fopen(categoryListPath);
categoryList = textscan(fid,'%s');
categoryList = categoryList{1};
fclose(fid);
%
nCategory = size(categoryList,1);
for iCategory =  1 : nCategory
    sampleDataFile = [(paths.rootDir),(params.dataSet),(paths.sampleDir),(categoryList{iCategory}),num2str(params.sampleSize),'.cat'];
    sampleData = load(sampleDataFile);
    fprintf('%s loaded\n',sampleDataFile);
    nVec = size(sampleData,2);
    nSample = 10000;
    rndSample = randsample(nVec,nSample);
    sampleData = sampleData(:,rndSample);
    tempTimeDir = strcat(num2str(floor(now*10000000)),'/');
    tempPath = strcat(paths.rootDir,paths.tempDir,tempTimeDir);
    if ~exist(tempPath,'dir')
        mkdir(tempPath);
    end
    tempDataPath = strcat(tempPath,'tempdata');
    if ~exist(tempDataPath,'file')
        dlmwrite(tempDataPath,sampleData,' ');
    end
    tempDataDimPath = strcat(tempPath,'tempdata_dim');
    dataDim = size(sampleData)';
    if ~exist(tempDataDimPath,'file')
        fid = fopen(tempDataDimPath,'w');
        fprintf(fid,'%s\n%s',num2str(dataDim(1)),num2str(dataDim(2)));
        fclose(fid);
    end
    tempCCFilePath = strcat(paths.rootDir,params.dataSet,paths.coclustDir,categoryList{iCategory},num2str(params.dictSize),params.dictType,num2str(params.sampleSize),params.ccType,num2str(params.rowClust),num2str(params.colClust),'.s');

    progArgs = sprintf(' -A %s -R %d -C %d -I d s %s -O c s 0 o %s',params.ccType,params.rowClust,params.colClust,tempDataPath,tempCCFilePath);
    cmd = strcat(params.progPath,params.prog,progArgs);
    % system call to cocluster linux program
    fprintf('%s\n','co-clustering...');
    system(cmd);    
    if exist(tempCCFilePath,'file')
        fprintf('%s written\n',tempCCFilePath);
        rmdir(tempPath,'s');    
    else
        fprintf('%s ERROR\n',tempCCFilePath);
    end    
end
end