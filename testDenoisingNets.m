function [cleanAudio,noisyAudio,denoisedAudioFullyConnected] = testDenoisingNets(ads,...
                            noise,denoiseNetFullyConnected,noisyMean,...
                            noisyStd,cleanMean,cleanStd,SNR,src,dest_path,spectro_file)
% testDenoisingNets: Test fully speech denoising networks.
%
% This function is used in SpeechDenoisingExample
%
% ads: audioDatastore containing files to test
% denoiseNetFullyConnected: Fully connected pre-trained network
% denoiseNetFullyConvolutional: Fully convolutional pre-trained network
% noisyMean: Mean of the noisy STFT vectors used in training
% noisyStd: Standard deviation of the noisy STFT vectors used in training
% cleanMean: Mean of the baseline STFT vectors used in training
% cleanStd: Standard deviation of the baseline STFT vectors used in training
%
% The example used the pre-trained networks saved in denoisenet.mat

% Copyright 2018 The MathWorks, Inc.


WindowLength = 512;
win          = hamming(WindowLength,'periodic');
Overlap      = round(0.75 * WindowLength);
FFTLength    = WindowLength;
NumFeatures  = FFTLength/2 + 1;
NumSegments  = 8;


%%
% Read the contents of a file from the datastore
[cleanAudio,info] = read(ads);
Fs = info.SampleRate;

[p, q] = rat(24000/25000);
cleanAudio = resample(cleanAudio, p, q);
Fs = 24000;
%%
% Make sure the audio length is a multiple of the sample rate converter
% decimation factor
D            = 3;
L            = floor(numel(cleanAudio)/D);
cleanAudio   = cleanAudio(1:L*D);

%%
% Convert the audio signal to 8 KHz:

cleanAudio   = src(cleanAudio);
reset(src)

%%
% In this testing tage, we will corrupt speech with washing machine noise
% not used in te training stage. Add noise to the speech signal such that
% SNR is 0 dB.
% noise   = audioread('D:\College_Stuff\Academics\Final Year Project\Speech Intelligibility\Datasets\MRT Library\bg_noise\bg_loops\firetruck_pump_panel.wav');
% noise = audioread("/home/shobansb/Himavanth/noises/n5.wav");
L       = floor(numel(noise)/D);
noise   = noise(1:L*D);
noise   = src(noise);
% reset(src)
%%
% Create a random noise segment from the washing machine noise vector
% randind      = randi(numel(noise) - numel(cleanAudio) , [1 1]);
% noiseSegment = noise(randind : randind + numel(cleanAudio) - 1);

%% Make the length of speech and noise as equal 
%-------------------------------------------------------------------------
ls = length(cleanAudio);
ln = length(noise);
if(ls >= ln)  % Make the length of speech and noise equal
    cleanAudio = cleanAudio(1:ln);
else
    noise = noise(1:ls);
end
  
%%
% % Add noise to the speech signal. The SNR is 0 dB.
% noisePower   = sum(noiseSegment.^2);
% cleanPower   = sum(cleanAudio.^2);
% noiseSegment = noiseSegment .* sqrt(cleanPower/noisePower);
% noisyAudio   = cleanAudio + noiseSegment;

%%
% SNR = 4.27;
change = 20*log10(std(cleanAudio)/std(noise))-SNR;
scalednoise = noise*10^(change/20);
noisyAudio = cleanAudio + scalednoise;

%%
% Use |spectrogram| to generate magnitude STFT vectors from the noisy audio signals:
noisySTFT  = stft(noisyAudio,'Window',win,'OverlapLength',Overlap, 'FFTLength',FFTLength);
noisySTFT  = noisySTFT(NumFeatures-1:end,:);
noisyPhase = angle(noisySTFT);
noisySTFT  = abs(noisySTFT);

%%
% Generate the 8-segment training predictor signals from the noisy STFT.
% The overlap between consecutive predictors is equal to 7 segments.
noisySTFT    = [noisySTFT(:,1:NumSegments-1) noisySTFT];
predictors = zeros( NumFeatures, NumSegments , size(noisySTFT,2) - NumSegments + 1);
for index     = 1 : size(noisySTFT,2) - NumSegments + 1
    predictors(:,:,index) = noisySTFT(:,index:index+NumSegments-1); 
end

%%
% Normalize the predictors by the mean and standard deviation
% computed in the training stage:
predictors(:) = (predictors(:) - noisyMean) / noisyStd;

%%
% Compute the denoised magnitude STFT by using |predict| with the two
% trained networks.
predictors = reshape(predictors,[NumFeatures, NumSegments,1,size(predictors,3)]);
STFTFullyConnected     = predict(denoiseNetFullyConnected , predictors);
% STFTFullyConvolutional = predict(denoiseNetFullyConvolutional , predictors);

%%
% Scale the outputs by the mean and standard deviation used in the
% training stage
STFTFullyConnected(:)     = cleanStd * STFTFullyConnected(:)     +  cleanMean;
% STFTFullyConvolutional(:) = cleanStd * STFTFullyConvolutional(:) +  cleanMean;

%%
% Convert the one-sided STFT to a centered STFT.
STFTFullyConnected     = STFTFullyConnected.' .* exp(1j*noisyPhase);
STFTFullyConnected     = [conj(STFTFullyConnected(end-1:-1:2,:)) ; STFTFullyConnected];
% STFTFullyConvolutional = squeeze(STFTFullyConvolutional) .* exp(1j*noisyPhase);
% STFTFullyConvolutional = [conj(STFTFullyConvolutional(end-1:-1:2,:)) ; STFTFullyConvolutional];

%%
% Compute the denoised speech signals. |istft| performs inverse STFT. Use
% the phase of the noisy STFT vectors.
denoisedAudioFullyConnected     = istft(STFTFullyConnected,'Window',win,'OverlapLength',Overlap, 'FFTLength',FFTLength,'ConjugateSymmetric',true);
% denoisedAudioFullyConvolutional = istft(STFTFullyConvolutional,'Window',win,'OverlapLength',Overlap, 'FFTLength',FFTLength,'ConjugateSymmetric',true);

%%
% % Plot the clean, noisy and denoised audio signals.
% z = figure;
% subplot(311)
% t = (1/Fs) * ( 0:numel(denoisedAudioFullyConnected)-1);
% plot(t,cleanAudio(1:numel(denoisedAudioFullyConnected)))
% plotsetting('Time(s)','Clean Speech')
% % title('Clean Speech')
% grid on
% subplot(312)
% plot(t,noisyAudio(1:numel(denoisedAudioFullyConnected)))
% plotsetting('Time(s)','Noisy Speech')
% % title('Noisy Speech')
% grid on
% subplot(313)
% plot(t,denoisedAudioFullyConnected)
% plotsetting('Time(s)','Denoised Speech')
% % title('Denoised Speech (Fully Connected Layers)')
% grid on
% subplot(414)
% plot(t,denoisedAudioFullyConvolutional)
% title('Denoised Speech (Convolutional Layers)')
% grid on
%xlabel('Time (s)')

%% Saving the spectrogram
% time_file = join([spectro_file '_timedomain'],"");
% png_name = join([time_file '.png'],"");
% eps_name = join([time_file '.eps'],"");
% saveas(z,png_name);
% saveas(z,eps_name,'epsc');

%%
% % Plot the clean, noisy and denoised spectrograms.
% h = figure;
% subplot(311)
% spectrogram(cleanAudio, win, Overlap, FFTLength,Fs);
% plotsetting('Time(s)','Clean Speech')
% % title('Clean Speech')
% grid on
% subplot(312)
% spectrogram(noisyAudio, win, Overlap, FFTLength,Fs);
% plotsetting('Time(s)','Noisy Speech')
% % title('Noisy Speech')
% grid on
% subplot(313)
% spectrogram(denoisedAudioFullyConnected, win, Overlap, FFTLength,Fs);
% plotsetting('Time(s)','Denoised Speech')
% % title('Denoised Speech (Fully Connected Layers)')
% grid on
% % subplot(414)
% % spectrogram(denoisedAudioFullyConvolutional, win, Overlap, FFTLength,Fs);
% % title('Denoised Speech (Convolutional Layers)')
% % grid on
% p = get(h,'Position');
% set(h,'Position',[p(1) 65 p(3) 800]);
% 
% %% Saving the spectrogram
% spectro_file = join([spectro_file '_spectrogram'],"");
% png_name = join([spectro_file '.png'],"");
% eps_name = join([spectro_file '.eps'],"");
% saveas(h,png_name);
% saveas(h,eps_name,'epsc');

%% Saving the file in desired location
% filename = info.FileName;
% filename = split(filename,'\');
% filename = filename{end};

% dest_path = '/home/himavanth/processed_speech/';
% dest_path = 'D:\College_Stuff\Academics\Final Year Project\Speech Intelligibility\Datasets\processed_speech\';
% file_path = join([dest_path filename],"");
% audiowrite(file_path,denoisedAudioFullyConnected,Fs/D)