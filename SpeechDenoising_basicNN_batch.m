%%
clearvars;
close all;
clc;

%%
% Specify the data location
datafolder = "C:\Users\himavanth_reddy19\Downloads\Project\34speaker_dataset";
% datafolder = "/home/himavanth/34speaker_dataset";
% datafolder = "D:\College_Stuff\Academics\Final Year Project\Speech Intelligibility\Datasets\34speaker_dataset";
% filename = 'testdata.xlsx';

% Use |audioDatastore| to create a datastore for all files in the dataset.
ads = audioDatastore(datafolder);
ads = shuffle(ads);
num_train_files = 50;
ads_Tr = subset(ads,1:num_train_files);

%%
% Define the sample rate converter used to convert the 48 kHz audio to 8 kHz.
windowLength = 512;
win          = hamming(windowLength,"periodic");
overlap      = round(0.75 * windowLength);
ffTLength    = windowLength;
inputFs      = 24e3;
fs           = 8e3;
numFeatures  = ffTLength/2 + 1;
numSegments = 8;

%Define the sample rate converter used to convert the 48 kHz audio to 16 kHz.
% src = dsp.SampleRateConverter("InputSampleRate",inputFs, ...
%                               "OutputSampleRate",fs);
              
src = dsp.SampleRateConverter("InputSampleRate",inputFs, ...
                              "OutputSampleRate",fs, ...
                              "Bandwidth",7920);

%%
inp_SNR = [-10, -5, 0, 5, 10];
% inp_SNR = [-10,-5];
% noise_num = ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12", "n13", "n14", "n15", "n16", "n17"];
noise_num = ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12", "n13"];
% noise_num = ["n1"];
parent_folder = 'C:\Users\himavanth_reddy19\Downloads\Project\temp\';

targets = tall([]);
predictors = tall([]);

noises_dir = 'C:\Users\himavanth_reddy19\Downloads\Project\noises\';

for nn=1:length(noise_num)

noise_path = join([noises_dir noise_num(nn) '.wav'],"");
noise = audioread(noise_path);

for ss=1:length(inp_SNR)
SNR = inp_SNR(ss);

%% 
T = tall(ads_Tr);
[targets_temp,predictors_temp] = cellfun(@(x)HelperGenerateSpeechDenoisingFeatures(x,noise,src,SNR),T,"UniformOutput",false);

if nn == 1 && ss == 1
    targets = targets_temp;
    predictos = predictors_temp;
else    
    targets = cat(1,targets,targets_temp);
    predictors = cat(1,predictors,predictors_temp);
end



end

end
%%
[targets,predictors] = gather(targets,predictors);

% [targets,predictors] = peach(gather,[targets,predictors],1:num_train_files);
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
miniBatchSize = 256;
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
% if (SNR > 0); letter = 'p' ; else; letter = 'n'; end
% snr_string = [letter num2str(abs(SNR)) 'snr'];
% name_of_matrix = join(['denoisenet_' num2str(1000) 'files_' snr_string '_grid_' noise_num(nn) '.mat'],"");
% folder_name = join(['denoisenet_' num2str(1000) 'files_' snr_string '_grid_' noise_num(nn)],"");
% folder_path = join([parent_folder folder_name '\'],"");
% mkdir(folder_path);
folder_path = '';
spectro_file = '';

%%
doTraining = true;
if doTraining
    [denoiseNetFullyConnected, nn_info] = trainNetwork(trainPredictors,trainTargets,layers,options);
    save('temp.mat','denoiseNetFullyConnected')
    
else
    s = load(name_of_matrix);
    denoiseNetFullyConnected = s.denoiseNetFullyConnected;
end

%%
% spectro_file = join([snr_string '_grid_' noise_num(nn)],"");
test_folder = 'C:\Users\himavanth_reddy19\Downloads\Project\34speaker_dataset';
ads = audioDatastore(test_folder);
ads_Ts = shuffle(ads);
% r = randi([1 34000],1,1)
ads_Ts = subset(ads,6000:6010);
% ads_Ts = shuffle(ads_Ts);
N = length(ads_Ts.Files);

% fs = 16000;
% snr_arr = zeros(1,N);
% STOI = zeros(1,N);
% NCM_Val = zeros(1,N);
% CSIIh = zeros(1,N);
% CSIIm = zeros(1,N);
% CSIIl = zeros(1,N);

noise_path = join([noises_dir 'n2' '.wav'],"");
noise = audioread(noise_path);
SNR = -10;
% 

for i = 1:N
 [cleanAudio,noisyAudio,denoisedAudioFullyConnected] = testDenoisingNets(ads_Ts,noise,denoiseNetFullyConnected,noisyMean,noisyStd,cleanMean,cleanStd,SNR,src,folder_path,spectro_file);
% 
denoisedAudioFullyConnected = double(denoisedAudioFullyConnected);
%  
% [p, q] = rat(16000/25000);
% cleanAudio = resample(cleanAudio, p, q);
% denoisedAudioFullyConnected = resample(denoisedAudioFullyConnected, p, q);
% 
 %-----------------Quality Measure-----------------------------
 %1.SNR Output
 snr_arr(i) = 10*log10((sum(cleanAudio(1:numel(denoisedAudioFullyConnected)).^2)/sum((cleanAudio(1:numel(denoisedAudioFullyConnected))-denoisedAudioFullyConnected).^2)));
%  
%  %2.PESQ Measure
%  audiowrite('adsTs.wav',cleanAudio,fs);
%  audiowrite('enhancedads_Ts.wav',denoisedAudioFullyConnected,fs);
%  PESQ = pesq('adsTs.wav', 'enhancedads_Ts.wav');
%  
%  %-----------------Intelligibility Measure-----------------------------
%  %3.STOI Measure
% 
% STOI(i) = stoi(cleanAudio(1:numel(denoisedAudioFullyConnected)), denoisedAudioFullyConnected, fs);
%   
% %4.CSII Measure
%  [CSIIh(i),CSIIm(i),CSIIl(i)]= CSII('adsTs.wav', 'enhancedads_Ts.wav');
%  
% %5. NCM Measure
%  NCM_val(i)= NCM('adsTs.wav', 'enhancedads_Ts.wav');
%  
% end
% 
% PESQ(PESQ(:)== Inf)=NaN;
% PESQAvg = nanmean(PESQ(:))
% snr_arr(find(snr_arr(:)== Inf))=NaN;
% snr_arr(find(snr_arr(:)== -Inf))=NaN;
SNRAvg = nanmean(snr_arr(:))
% STOI(find(STOI(:)==Inf))=NaN;
% STOIAvg = nanmean(STOI(:))
% CSIIh(find(CSIIh(:)==Inf))=NaN;
% CSIIhAvg = nanmean(CSIIh(:))
% NCM_val(find(NCM_val(:)==Inf))=NaN;
% NCMAvg = nanmean(NCM_val(:))
% 
% 
% zeroth_cell = 'A';
% noise_cell = [zeroth_cell num2str(nn+2)];
% 
% first_cell = char(zeroth_cell + ss);
% 
% snr_cell = [first_cell num2str(nn+2)];
% pesq_cell = [char(first_cell + 5) num2str(nn+2)];
% stoi_cell = [char(first_cell + 10) num2str(nn+2)];
% csii_cell = [char(first_cell + 15) num2str(nn+2)];
% ncm_cell = [char(first_cell + 20) num2str(nn+2)];
% 
% writematrix(noise_num(nn),filename,'Sheet',1,'Range',noise_cell);
% writematrix(SNRAvg,filename,'Sheet',1,'Range',snr_cell);
% writematrix(PESQAvg,filename,'Sheet',1,'Range',pesq_cell);
% writematrix(STOIAvg,filename,'Sheet',1,'Range',stoi_cell);
% writematrix(CSIIhAvg,filename,'Sheet',1,'Range',csii_cell);
% writematrix(NCMAvg,filename,'Sheet',1,'Range',ncm_cell);
% 

end