%%
clearvars;
close all;
clc;

%%
% Specify the data location
% datafolder = "D:\College_Stuff\Academics\Final Year Project\Speech Intelligibility\Datasets\Unprocessed_MRT_1200files_Dataset\C01";
% datafolder = "/home/himavanth/34speaker_dataset";
datafolder = "D:\College_Stuff\Academics\Final Year Project\Speech Intelligibility\Datasets\34speaker_dataset";
filename = 'avg_pesq.xlsx';
second_file = 'max_pesq.xlsx';

% Use |audioDatastore| to create a datastore for all files in the dataset.
ads = audioDatastore(datafolder);
ads = shuffle(ads);
num_train_files = 100;
ads_Tr = subset(ads,1:num_train_files);

%%
inp_SNR = [-10, -5, 0, 5, 10];
% noise_num = ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12", "n13", "n14", "n15", "n16", "n17"];
noise_num = ["n6","n11"];
%%
for nn=1:length(noise_num)
noises_dir = 'D:\College_Stuff\Academics\Final Year Project\Speech Intelligibility\Datasets\noises\';

noise_num(nn)
noise_path = join([noises_dir noise_num(nn) '.wav'],"");
noise = audioread(noise_path);


for ss=1:length(inp_SNR)
%%
SNR = inp_SNR(ss)
PESQ = zeros(1,num_train_files);
snr_arr = zeros(1,num_train_files);
fs = 16000;

%% Make the length of speech and noise as equal 
%-------------------------------------------------------------------------
for i=1:num_train_files
[speech,info] = read(ads_Tr);

ls = length(speech);
ln = length(noise);
if(ls >= ln)  % Make the length of speech and noise equal
    speech = speech(1:ln);
else
    noise = noise(1:ls);
end


%% Scale the noise such that speech+noise = noisyspeech at the desired SNR
% -----------------------------------------------------------------------

% Desired Signal to Noise Ratio
change = 20*log10(std(speech)/std(noise))-SNR;
scalednoise = noise*10^(change/20);
noisyspeech = speech + scalednoise;

%%
[p, q] = rat(16000/25000);
speech = resample(speech, p, q);
noisyspeech = resample(noisyspeech, p, q);

audiowrite('adsTs.wav',speech,fs);
audiowrite('NS_Ts.wav',noisyspeech,fs);
PESQ(i) = pesq('adsTs.wav', 'NS_Ts.wav');
% snr_arr(i) = 10*log10((sum(speech(1:numel(noisyspeech)).^2)/sum((speech(1:numel(noisyspeech))-noisyspeech).^2)));
end
PESQAvg = mean(PESQ);
reset(ads_Tr);
% SNRAvg = mean(snr_arr)


zeroth_cell = 'A';
noise_cell = [zeroth_cell num2str(nn)];
pesq_cell = [char(zeroth_cell + ss) num2str(nn)];


writematrix(noise_num(nn),second_file,'Sheet',1,'Range',noise_cell);
writematrix(max(PESQ),second_file,'Sheet',1,'Range',pesq_cell);

writematrix(noise_num(nn),filename,'Sheet',1,'Range',noise_cell);
writematrix(PESQAvg,filename,'Sheet',1,'Range',pesq_cell);

end

end