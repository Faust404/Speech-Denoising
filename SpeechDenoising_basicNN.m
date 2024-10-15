%%
clearvars;
close all;
clc;

%%
% Specify the data location
% datafolder = "D:\College_Stuff\Academics\Final Year Project\Speech Intelligibility\Datasets\Unprocessed_MRT_1200files_Dataset\C01";
% datafolder = "/home/himavanth/34speaker_dataset";
datafolder = "D:\College_Stuff\Academics\Final Year Project\Speech Intelligibility\Datasets\34speaker_dataset";
filename = 'testdata.xlsx';

% Use |audioDatastore| to create a datastore for all files in the dataset.
ads = audioDatastore(datafolder);
ads = shuffle(ads);
num_train_files = 150;
ads_Tr = subset(ads,1:num_train_files);

%%
% Define the sample rate converter used to convert the 48 kHz audio to 8 kHz.
windowLength = 256;
win          = hamming(windowLength,"periodic");
overlap      = round(0.75 * windowLength);
ffTLength    = windowLength;
% inputFs      = 24e3;
% fs           = 8e3;
inputFs      = 25e3;
fs           = 25e3;
numFeatures  = ffTLength/2 + 1;
numSegments = 8;

%Define the sample rate converter used to convert the 48 kHz audio to 16 kHz.
src = dsp.SampleRateConverter("InputSampleRate",inputFs, ...
                              "OutputSampleRate",fs);
              
% src = dsp.SampleRateConverter("InputSampleRate",inputFs, ...
%                               "OutputSampleRate",fs, ...
%                               "Bandwidth",7920);


%%
% noise = audioread("D:\College_Stuff\Academics\Final Year Project\Speech Intelligibility\Datasets\MRT Library\bg_noise\bg_loops\firetruck_pump_panel.wav");
% noises_dir = '/home/himavanth/noises/';
noises_dir = 'D:\College_Stuff\Academics\Final Year Project\Speech Intelligibility\Datasets\noises\';
noise_num = 'n1';
noise = audioread([noises_dir noise_num '.wav']);
SNR = -5;


%% 
T = tall(ads_Tr);
[targets,predictors] = cellfun(@(x)HelperGenerateSpeechDenoisingFeatures(x,noise,src,SNR),T,"UniformOutput",false);

%%
[targets,predictors] = gather(targets,predictors);

%% 
predictors    = cat(3,predictors{:});
targets       = cat(2,targets{:});
noisyMean     = mean(predictors(:));
noisyStd      = std(predictors(:));
predictors(:) = (predictors(:) - noisyMean)/noisyStd;
cleanMean     = mean(targets(:));
cleanStd      = std(targets(:));
targets(:)    = (targets(:) - cleanMean)/cleanStd;

%%
predictors = reshape(predictors,size(predictors,1),size(predictors,2),1,size(predictors,3));
targets    = reshape(targets,1,1,size(targets,1),size(targets,2));

%%
inds               = randperm(size(predictors,4));
L                  = round(0.99 * size(predictors,4));
trainPredictors    = predictors(:,:,:,inds(1:L));
trainTargets       = targets(:,:,:,inds(1:L));
validatePredictors = predictors(:,:,:,inds(L+1:end));
validateTargets    = targets(:,:,:,inds(L+1:end));

%%
layers = [
    imageInputLayer([numFeatures,numSegments])
    fullyConnectedLayer(1024)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(1024)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numFeatures)
    regressionLayer
    ];

%%
miniBatchSize = 32;
options = trainingOptions("adam", ...
    "MaxEpochs",3, ...
    "InitialLearnRate",1e-5,...
    "MiniBatchSize",miniBatchSize, ...
    "Shuffle","every-epoch", ...
    "Plots","training-progress", ...
    "Verbose",false, ...
    "ValidationFrequency",floor(size(trainPredictors,4)/miniBatchSize), ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.9, ...
    "LearnRateDropPeriod",1, ...
    "ValidationData",{validatePredictors,validateTargets});

%%
if (SNR > 0); letter = 'p' ; else; letter = 'n'; end
snr_string = [letter num2str(abs(SNR)) 'snr'];
name_of_matrix = join(['denoisenet_' num2str(num_train_files) 'files_' snr_string '_grid_' noise_num '.mat'],"");


%%
doTraining = true;
if doTraining
    [denoiseNetFullyConnected, nn_info] = trainNetwork(trainPredictors,trainTargets,layers,options);
    save(name_of_matrix,'denoiseNetFullyConnected')
    
else
    s = load("BasicNN_Specific_SNR_GRID/denoisenet_1000files_p4.88snr_grid_n1.mat");
    denoiseNetFullyConnected = s.denoiseNetFullyConnected;
end

%%
ads_Ts = subset(ads,2001:2100);
ads_Ts = shuffle(ads_Ts);
N = length(ads_Ts.Files);

% fs = 8000;
fs = 16000;
snr_arr = zeros(1,N);
STOI = zeros(1,N);
NCM_Val = zeros(1,N);
CSIIh = zeros(1,N);
CSIIm = zeros(1,N);
CSIIl = zeros(1,N);


for i = 1:N
 [cleanAudio,noisyAudio,denoisedAudioFullyConnected] = testDenoisingNets(ads_Ts,noise,denoiseNetFullyConnected,noisyMean,noisyStd,cleanMean,cleanStd,SNR,src);

denoisedAudioFullyConnected = double(denoisedAudioFullyConnected);
 
[p, q] = rat(16000/25000);
cleanAudio = resample(cleanAudio, p, q);
denoisedAudioFullyConnected = resample(denoisedAudioFullyConnected, p, q);

 %-----------------Quality Measure-----------------------------
 %1.SNR Output
 snr_arr(i) = 10*log10((sum(cleanAudio(1:numel(denoisedAudioFullyConnected)).^2)/sum((cleanAudio(1:numel(denoisedAudioFullyConnected))-denoisedAudioFullyConnected).^2)));
 
 %2.PESQ Measure
 audiowrite('adsTs.wav',cleanAudio,fs);
 audiowrite('enhancedads_Ts.wav',denoisedAudioFullyConnected,fs);
 PESQ = pesq('adsTs.wav', 'enhancedads_Ts.wav');
 
 %-----------------Intelligibility Measure-----------------------------
 %3.STOI Measure

STOI(i) = stoi(cleanAudio(1:numel(denoisedAudioFullyConnected)), denoisedAudioFullyConnected, fs);
  
%4.CSII Measure
 [CSIIh(i),CSIIm(i),CSIIl(i)]= CSII('adsTs.wav', 'enhancedads_Ts.wav');
 
%5. NCM Measure
 NCM_val(i)= NCM('adsTs.wav', 'enhancedads_Ts.wav');
 
end

PESQ(PESQ(:)== Inf)=NaN;
PESQAvg = nanmean(PESQ(:))
snr_arr(find(snr_arr(:)== Inf))=NaN;
snr_arr(find(snr_arr(:)== -Inf))=NaN;
SNRAvg = nanmean(snr_arr(:))
STOI(find(STOI(:)==Inf))=NaN;
STOIAvg = nanmean(STOI(:))
CSIIh(find(CSIIh(:)==Inf))=NaN;
CSIIhAvg = nanmean(CSIIh(:))
NCM_val(find(NCM_val(:)==Inf))=NaN;
NCMAvg = nanmean(NCM_val(:))
