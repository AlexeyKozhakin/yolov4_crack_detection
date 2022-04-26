close all; clear all; clc;

data1 = load('data_crack2.mat');
%vehicleDataset = data.vehicleDataset;
%vehicleDataset = [data1.gTruth.DataSource.Source data1.gTruth.LabelData];
vehicleDataset.imageFilename=[data1.gTruth.DataSource.Source];
vehicleDataset.crack=[data1.gTruth.LabelData.crack];
vehicleDataset = struct2table(vehicleDataset);


% Display first few rows of the data set.
vehicleDataset(1:4,:)

% % Add the fullpath to the local vehicle data folder.
% vehicleDataset.imageFilename = fullfile(pwd,vehicleDataset.imageFilename);

rng("default");
shuffledIndices = randperm(height(vehicleDataset));
%idx = floor(0.6 * length(shuffledIndices) );
idx = 64;

trainingIdx = 1:idx;
trainingDataTbl = vehicleDataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : length(shuffledIndices);
validationDataTbl = vehicleDataset(shuffledIndices(validationIdx),:);


imdsTrain = imageDatastore(trainingDataTbl{:,"imageFilename"});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,"crack"));

imdsValidation = imageDatastore(validationDataTbl{:,"imageFilename"});
bldsValidation = boxLabelDatastore(validationDataTbl(:,"crack"));



trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);




data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,"Rectangle",bbox,'LineWidth',15);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

reset(trainingData);

inputSize = [96 96 3];

className = "crack";

rng("default")
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputSize));
numAnchors = 9;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);

area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");

anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    anchors(7:9,:)
    };

detector = yolov4ObjectDetector("csp-darknet53-coco",className,anchorBoxes,InputSize=inputSize);

augmentedTrainingData = transform(trainingData,@augmentData);


augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},"rectangle",data{2},'LineWidth',15);
    reset(augmentedTrainingData);
end
figure
montage(augmentedData,BorderSize=10)

parpool('local',1)

options = trainingOptions("adam",...
    GradientDecayFactor=0.9,...
    SquaredGradientDecayFactor=0.999,...
    InitialLearnRate=0.001,...
    LearnRateSchedule="none",...
    MiniBatchSize=16,...
    L2Regularization=0.0005,...
    MaxEpochs=50,...
    BatchNormalizationStatistics="moving",...
    DispatchInBackground=true,...
    ResetInputNormalization=false,...
    Shuffle="every-epoch",...
    VerboseFrequency = 1, ...
    ValidationFrequency = 1, ...
    ValidationData=validationData);



    % Train the YOLO v4 detector.
[detector,info] = trainYOLOv4ObjectDetector(augmentedTrainingData,detector,options);

I = imread('C:\data2\t3.jpeg');
[bboxes,scores,labels] = detect(detector,I);

I = insertObjectAnnotation(I,"rectangle",bboxes,scores,'LineWidth',5);
figure
imshow(I)

