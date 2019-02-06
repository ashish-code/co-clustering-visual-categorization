% function to calculate coefficients for subspacedictionary
function coclustsubspaceCoeff(dataSet,dictSize,colClust,ccType)
LASTN = maxNumCompThreads('automatic');
fprintf('%s\t%d\n','numThreads',LASTN);
dictType = 'universal';
%
% initialize matlab
cdir = pwd;
cd ~;
startup;
cd (cdir);
%
rootDir = '/vol/vssp/diplecs/ash/Data/';
categoryListFileName = 'categoryList.txt';
dictDir = '/Dictionary/';
imageListDir = '/ImageLists/';
coeffDir = '/Coeff/';
% read the category list in the dataset
categoryListPath = [(rootDir),(dataSet),'/',(categoryListFileName)];
fid = fopen(categoryListPath,'r');
categoryList = textscan(fid,'%s');
categoryList = categoryList{1};
fclose(fid);
%
nCategory = size(categoryList,1);
listSizes = 30;
nListSizes = max(size(listSizes));
%
for iCategory = 1 : nCategory      
    subspacedictFilePath = strcat(rootDir,dataSet,dictDir,dataSet,num2str(dictSize),dictType,'kmeans',ccType,colClust,'.dict');
    dict = load(subspacedictFilePath);
    dict = dict';
    if ismember(dataSet,['Scene15','Caltech101','Caltech256'])
        coeffCatDir = [(rootDir),(dataSet),(coeffDir),categoryList{iCategory}];
        if exist(coeffCatDir,'dir') ~= 7
            mkdir(coeffCatDir)
        end
    end
    %
    for iListSize = 1 : nListSizes
        listTrainPosFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Train',num2str(listSizes(iListSize)),'.pos'];
        listValPosFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Val',num2str(listSizes(iListSize)),'.pos'];
        listTrainNegFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Train',num2str(listSizes(iListSize)),'.neg'];
        listValNegFile = [(rootDir),(dataSet),(imageListDir),categoryList{iCategory},'Val',num2str(listSizes(iListSize)),'.neg'];
        %        
        fid = fopen(listTrainPosFile,'r');
        listTrainPos = textscan(fid,'%s');
        fclose(fid);
        listTrainPos = listTrainPos{1};
        %
        fid = fopen(listValPosFile,'r');
        listValPos = textscan(fid,'%s');
        fclose(fid);
        listValPos = listValPos{1};
        %
        fid = fopen(listTrainNegFile,'r');
        listTrainNeg = textscan(fid,'%s');
        fclose(fid);
        listTrainNeg = listTrainNeg{1};
        %
        fid = fopen(listValNegFile,'r');
        listValNeg = textscan(fid,'%s');
        fclose(fid);
        listValNeg = listValNeg{1};
        %
        nListTrainPos = size(listTrainPos,1);
        nListValPos = size(listValPos,1);
        nListTrainNeg = size(listTrainNeg,1);
        nListValNeg = size(listValNeg,1);         
        % Train ; Pos
        for iter = 1 : nListTrainPos
            imageName = listTrainPos{iter};
            callSubspaceVQEnc(imageName,dict,dataSet,dictType,dictSize,ccType,colClust);            
        end        
        % Val ; Pos
        for iter = 1 : nListValPos
            imageName = listValPos{iter};
            callSubspaceVQEnc(imageName,dict,dataSet,dictType,dictSize,ccType,colClust);           
        end        
        % Train ; Neg
        for iter = 1 : nListTrainNeg
            imageName = listTrainNeg{iter};
            callSubspaceVQEnc(imageName,dict,dataSet,dictType,dictSize,ccType,colClust);           
        end        
        % Val ; Neg
        for iter = 1 : nListValNeg
            imageName = listValNeg{iter};
            callSubspaceVQEnc(imageName,dict,dataSet,dictType,dictSize,ccType,colClust);         
        end
    end
end

end

function callSubspaceVQEnc(imageName,dict,dataSet,dictType,dictSize,ccType,colClust)
rootDir = '/vol/vssp/diplecs/ash/Data/';
coeffDir = '/Coeff/';
dsiftDir = '/DSIFT/';
%
coeffFilePathAvg = [(rootDir),(dataSet),(coeffDir),imageName,num2str(dictSize),(dictType),num2str(colClust),ccType,'.ccss'];
if exist(coeffFilePathAvg,'file')
    return;
end
%
imageFilePath = [(rootDir),(dataSet),(dsiftDir),(imageName),'.dsift'];
imageData = load(imageFilePath);
imageData = imageData(3:130,:);
imageData = imageData';
nVec = size(imageData,1);
%
coeff = zeros(1,dictSize);
% for each vector in an image
for i = 1 : nVec
    % for each dictionary element
    dd = zeros(1,dictSize);
    for j = 1 : dictSize
        ivec = imageData(i,:);
        ivec(dict(j,:)==0)=0;
        dd(j) = norm(ivec-dict(j,:));
    end
    didx = find(dd == min(dd));
    coeff(didx) = coeff(didx) + 1;
end
coeff = coeff./sum(coeff);

dlmwrite(coeffFilePathAvg,coeff,'delimiter',',');
fprintf('%s\n',coeffFilePathAvg);

end