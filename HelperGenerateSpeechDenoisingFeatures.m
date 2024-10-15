function [targets,predictors] = HelperGenerateSpeechDenoisingFeatures(audio,noise,src,SNR)
% HelperGenerateSpeechDenoisingFeatures: Get target and predictor STFT
% signals for speech denoising.
% audio: Input audio signal
% noise: Input noise signal
% src:   Sample rate converter

% Copyright 2018 The MathWorks, Inc.

WindowLength = 512;
win          = hamming(WindowLength,'periodic');
Overlap      = round(0.75 * WindowLength);
FFTLength    = WindowLength;
NumFeatures  = FFTLength/2 + 1;
NumSegments  = 8;

[p, q] = rat(24000/25000);
audio = resample(audio, p, q);

D            = 3; % Decimation factor
L            = floor( numel(audio)/D);
audio        = audio(1:D*L);


audio   = src(audio);
reset(src);

%%
% randind      = randi(numel(noise) - numel(audio) , [1 1]);
% noiseSegment = noise(randind : randind + numel(audio) - 1);
% 
% noisePower   = sum(noiseSegment.^2);
% cleanPower   = sum(audio.^2);
% noiseSegment = noiseSegment .* sqrt(cleanPower/noisePower);
% noisyAudio   = audio + noiseSegment;
%% 
L       = floor(numel(noise)/D);
noise   = noise(1:L*D);
noise   = src(noise);
reset(src)

%% Make the length of speech and noise as equal 
%-------------------------------------------------------------------------
ls = length(audio);
ln = length(noise);
if(ls >= ln)  % Make the length of speech and noise equal
    audio = audio(1:ln);
else
    noise = noise(1:ls);
end
  
%%
% SNR = 4.95;
change = 20*log10(std(audio)/std(noise))-SNR;
scalednoise = noise*10^(change/20);
noisyAudio = audio + scalednoise;

%%
cleanSTFT = stft(audio, 'Window',win, 'OverlapLength', Overlap, 'FFTLength',FFTLength);
cleanSTFT = abs(cleanSTFT(NumFeatures-1:end,:));
noisySTFT = stft(noisyAudio, 'Window',win, 'OverlapLength', Overlap, 'FFTLength',FFTLength);
noisySTFT = abs(noisySTFT(NumFeatures-1:end,:));

noisySTFTAugmented    = [noisySTFT(:,1:NumSegments-1) noisySTFT];
 
STFTSegments = zeros( NumFeatures, NumSegments , size(noisySTFTAugmented,2) - NumSegments + 1);
for index     = 1 : size(noisySTFTAugmented,2) - NumSegments + 1
    STFTSegments(:,:,index) = noisySTFTAugmented(:,index:index+NumSegments-1);
end

targets    = cleanSTFT;
predictors = STFTSegments;