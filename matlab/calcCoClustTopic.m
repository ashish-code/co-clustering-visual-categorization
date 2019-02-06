% calculate bregman coclustering for list of training image coefficients
% for each dictionary type
function calcCoClustTopic(dataSet,dictType,dictSize,sampleSize,algo,algoParam,method,rowClust,colClust,ccType)
% function calcCoClustTopic(dataSet,dictType,dictSize,sampleSize)
% dataSet : VOC2006,VOC2007,VOC2010,Scene15,Caltech101,Caltech256
% dictType: universal, categorical, balanced
% dictSize: 1000
% sampleSize: 100000
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
paths.tempDir = 'Temp/';
%---------------------------------------------------------------------
params.dataSet = dataSet;
params.dictType = dictType;
params.dictSize = dictSize;
params.sampleSize = sampleSize;
params.algo = algo;
params.algoParam = algoParam;
params.rowClust = rowClust;
params.colClust = colClust;
params.ccType = ccType;
params.categoryListFileName = 'categoryList.txt';
params.imageListDir = '/ImageLists/';
params.progPath = '/vol/vssp/diplecs/ash/code/cocluster/';
params.prog = 'cocluster-linux';
params.listSize = 30;
params.method = method;
%

params.nNonZero = params.dictSize;

%---------------------------------------------------------------------
% the pipeline depends extensively upon the dictionary type
if strcmp(dictType,'universal')
    callCoClustTopicUniversal(params,paths);
elseif strcmp(dictType,'categorical')
    callCoClustTopicCategorical(params,paths);
elseif strcmp(dictType,'balanced')
    callCoClustTopicBalanced(params,paths);
end
%---------------------------------------------------------------------

end

function callCoClustTopicUniversal(params,paths)
% read the category list in the dataset
categoryListPath = [(paths.rootDir),(params.dataSet),'/',(params.categoryListFileName)];
fid = fopen(categoryListPath,'r');
categoryList = textscan(fid,'%s');
categoryList = categoryList{1};
fclose(fid);
nCategory = size(categoryList,1);
%
for iCategory = 1 : nCategory
    fprintf('%s\n',categoryList{iCategory});
    listTrainPosFile = [(paths.rootDir),(params.dataSet),(params.imageListDir),categoryList{iCategory},'Train',num2str(params.listSize),'.pos'];
    listTrainNegFile = [(paths.rootDir),(params.dataSet),(params.imageListDir),categoryList{iCategory},'Train',num2str(params.listSize),'.neg'];
    fid = fopen(listTrainPosFile,'r');listTrainPos = textscan(fid,'%s');fclose(fid);
    listTrainPos = listTrainPos{1};
    fid = fopen(listTrainNegFile,'r');listTrainNeg = textscan(fid,'%s');fclose(fid);
    listTrainNeg = listTrainNeg{1};
    nListTrainPos = size(listTrainPos,1);
    nListTrainNeg = size(listTrainNeg,1);
    FTrainPosAvg = ones(nListTrainPos,params.dictSize);
    FTrainNegAvg = zeros(nListTrainNeg,params.dictSize);
    
    % Train ; Pos
    for iter = 1 : nListTrainPos
        imageName = listTrainPos{iter};
        coeffFilePathAvg = [(paths.rootDir),(params.dataSet),(paths.coeffDir),imageName,num2str(params.dictSize),(params.dictType),num2str(params.sampleSize),(params.algo),num2str(params.algoParam),(params.method),num2str(params.nNonZero),'.avg'];
        try
            Favg = dlmread(coeffFilePathAvg,',');
        catch err
            disp(err.identifier);
            Favg = zeros(1,params.dictSize);
        end
        if size(Favg,1) > size(Favg,2)
            Favg = Favg';
        end                 
        FTrainPosAvg(iter,1:params.dictSize) = Favg;            
    end
    % Train ; Neg
    for iter = 1 : nListTrainNeg
        imageName = listTrainNeg{iter};
        coeffFilePathAvg = [(paths.rootDir),(params.dataSet),(paths.coeffDir),imageName,num2str(params.dictSize),(params.dictType),num2str(params.sampleSize),(params.algo),num2str(params.algoParam),(params.method),num2str(params.nNonZero),'.avg'];
        try
            Favg = dlmread(coeffFilePathAvg,',');
        catch err
            disp(err.identifier);
            disp(coeffFilePathAvg);
            Favg = zeros(1,params.dictSize);
        end
        if size(Favg,1) > size(Favg,2)
            Favg = Favg';
        end           
        FTrainNegAvg(iter,1:params.dictSize) = Favg;            
    end
    FTrainAvg = [FTrainPosAvg;FTrainNegAvg];
    tempTimeDir = strcat(num2str(floor(now*100000000)),'/');
    tempPath = strcat(paths.rootDir,paths.tempDir,tempTimeDir);
    if ~exist(tempPath,'dir')
        mkdir(tempPath);
    end
    tempDataPath = strcat(tempPath,'tempdata');
    if ~exist(tempDataPath,'file')
        dlmwrite(tempDataPath,FTrainAvg,' ');
    end
    tempDataDimPath = strcat(tempPath,'tempdata_dim');
    dataDim = size(FTrainAvg)';
    if ~exist(tempDataDimPath,'file')
        fid = fopen(tempDataDimPath,'w');
        fprintf(fid,'%s\n%s',num2str(dataDim(1)),num2str(dataDim(2)));
        fclose(fid);
    end
    tempCCFilePath = strcat(paths.rootDir,params.dataSet,paths.coclustDir,categoryList{iCategory},num2str(params.dictSize),params.dictType,num2str(params.sampleSize),params.algo,num2str(params.algoParam),params.method,params.ccType,num2str(params.rowClust),num2str(params.colClust),'.t');
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

function callCoClustTopicCategorical(params,paths)
% read the category list in the dataset
categoryListPath = [(paths.rootDir),(params.dataSet),'/',(params.categoryListFileName)];
fid = fopen(categoryListPath,'r');
categoryList = textscan(fid,'%s');
categoryList = categoryList{1};
fclose(fid);
nCategory = size(categoryList,1);
%
for iCategory = 1 : nCategory
    fprintf('%s\n',categoryList{iCategory});
    listTrainPosFile = [(paths.rootDir),(params.dataSet),(params.imageListDir),categoryList{iCategory},'Train',num2str(params.listSize),'.pos'];
    fid = fopen(listTrainPosFile,'r');listTrainPos = textscan(fid,'%s');fclose(fid);
    listTrainPos = listTrainPos{1};
    nListTrainPos = size(listTrainPos,1);
    FTrainPosAvg = ones(nListTrainPos,params.dictSize);
    % Train ; Pos
    for iter = 1 : nListTrainPos
        imageName = listTrainPos{iter};
        coeffFilePathAvg = [(paths.rootDir),(params.dataSet),(paths.coeffDir),imageName,num2str(params.dictSize),(params.dictType),num2str(params.sampleSize),(params.algo),num2str(params.algoParam),(params.method),num2str(params.nNonZero),'.avg'];
        try
            Favg = dlmread(coeffFilePathAvg,',');
        catch err
            disp(err.identifier);
            Favg = zeros(1,params.dictSize);
        end
        if size(Favg,1) > size(Favg,2)
            Favg = Favg';
        end                 
        FTrainPosAvg(iter,1:params.dictSize) = Favg;            
    end
    FTrainAvg = FTrainPosAvg;
    tempTimeDir = strcat(num2str(floor(now*1000000)),'/');
    tempPath = strcat(paths.rootDir,paths.tempDir,tempTimeDir);
    if ~exist(tempPath,'dir')
        mkdir(tempPath);
    end
    tempDataPath = strcat(tempPath,'tempdata');
    if ~exist(tempDataPath,'file')
        dlmwrite(tempDataPath,FTrainAvg,' ');
    end
    tempDataDimPath = strcat(tempPath,'tempdata_dim');
    dataDim = size(FTrainAvg)';
    if ~exist(tempDataDimPath,'file')
        fid = fopen(tempDataDimPath,'w');
        fprintf(fid,'%s\n%s',num2str(dataDim(1)),num2str(dataDim(2)));
        fclose(fid);
    end
    tempCCFilePath = strcat(paths.rootDir,params.dataSet,paths.coclustDir,categoryList{iCategory},num2str(params.dictSize),params.dictType,num2str(params.sampleSize),params.algo,num2str(params.algoParam),params.method,params.ccType,num2str(params.rowClust),num2str(params.colClust),'.t');
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

function callCoClustTopicBalanced(params,paths)

end