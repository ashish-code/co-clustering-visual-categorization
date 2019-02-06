function coclustSubspaceDict(dataSet,dictSize,rowClust,colClust,ccType)
dictType = 'universal';
sampleSize = 100000;

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

% ccfilepath
tempCCFilePath = strcat(paths.rootDir,params.dataSet,paths.coclustDir,params.dataSet,num2str(params.dictSize),params.dictType,num2str(params.sampleSize),params.ccType,num2str(params.rowClust),num2str(params.colClust),'.s');
%---------------------------------------------------------------------
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

ccFilePath = strcat(paths.rootDir,params.dataSet,paths.coclustDir,params.dataSet,num2str(params.dictSize),params.dictType,num2str(params.sampleSize),params.ccType,num2str(params.rowClust),num2str(params.colClust),'.s');
disp(ccFilePath);
try
    ccfid = fopen(ccFilePath);
    
catch err
    fprintf('%s, %s','unable to open ',ccFilePath);
    fprintf('%s\n',err.identifier);
    return;
end

% rowcc = fgetl(ccfid);
colcc = fgetl(ccfid);
fclose(ccfid);

% rowcc = textscan(rowcc,'%d ');
colcc = textscan(colcc,'%d ');

% rowcc = rowcc{1};
colcc = colcc{1};

% rowcc = rowcc+1;
colcc = colcc+1;

% [rowSort,rowIdx] = sort(rowcc);
[colSort,colIdx] = sort(colcc);

disp(colSort')
disp(colIdx');

colUnique = unique(colSort);

dictFilePath = strcat(paths.rootDir,params.dataSet,paths.dictDir,params.dataSet,num2str(params.dictSize),params.dictType,num2str(params.sampleSize),'kmeans','.dict');

if exist(dictFilePath,'file')
    dict = dlmread(dictFilePath,',');
    dict = dict';
else
    fprintf('%s\n','computing dictionary...');
    opts = statset('MaxIter',20);    
    [~, dict] = kmeans(sampleData,dictSize,'Start','cluster','EmptyAction','singleton','Options',opts);          
    dlmwrite(dictDataFile,dict','delimiter',',');
end

dict = dict(:,colIdx);


% find the maximum size
% colSort ; colUnique
nSubspace = max(size(colUnique));

% for each dictionary element, each vector in the dictionary
dictsubspace = zeros(size(dict,1),params.colClust);
for iDict = 1 : params.dictSize
    % compute
    dvec = dict(iDict,:);
    %
    normvec = zeros(nSubspace,1);
    for iSS = 1 : nSubspace
        dvecss = dvec(find(colSort == colUnique(iSS)));
        normvec(iSS) = norm(dvecss,2);
        dictsubspace(iDict,iSS) = normvec(iSS);
    end
end

figure(1),imshow(imcomplement(uint8(dict')));
figure(2),imshow(imcomplement(uint8(dictsubspace')));
figure(3),imshow(imcomplement(uint8(dictpca')));

pause(20);
end

