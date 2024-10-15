%%
clearvars;
close all;
clc;

%%
% Specify the data location
% datafolder = "D:\College_Stuff\Academics\Final Year Project\Speech Intelligibility\Datasets\Unprocessed_MRT_1200files_Dataset\C01";
% datafolder = "/home/himavanth/34speaker_dataset";
datafolder = "D:\College_Stuff\Academics\Final Year Project\Speech Intelligibility\Datasets\34speaker_dataset";
filename = 'csii_ncm_stoi_avg.xlsx';
second_file = 'csii_ncm_stoi_max.xlsx';

% Use |audioDatastore| to create a datastore for all files in the dataset.
ads = audioDatastore(datafolder);
ads = shuffle(ads);
num_train_files = 100;
ads_Tr = subset(ads,1:num_train_files);

%%
inp_SNR = [-10, -5, 0, 5, 10];
noise_num = ["n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "n10", "n11", "n12", "n13", "n14", "n15", "n16", "n17"];

%%
for nn=1:length(noise_num)
noises_dir = 'D:\College_Stuff\Academics\Final Year Project\Speech Intelligibility\Datasets\noises\';

noise_path = join([noises_dir noise_num(nn) '.wav'],"");
noise = audioread(noise_path);


for ss=1:length(inp_SNR)
%%
SNR = inp_SNR(ss);
STOI = zeros(1,num_train_files);
NCM_Val = zeros(1,num_train_files);
CSIIh = zeros(1,num_train_files);
CSIIm = zeros(1,num_train_files);
CSIIl = zeros(1,num_train_files);
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
STOI(i) = stoi(speech(1:numel(noisyspeech)), noisyspeech, fs);
  
%4.CSII Measure
[CSIIh(i),CSIIm(i),CSIIl(i)]= CSII('adsTs.wav', 'NS_Ts.wav');
 
%5. NCM Measure
NCM_val(i)= NCM('adsTs.wav', 'NS_Ts.wav');

end
STOIAvg = mean(STOI);
CSIIhAvg = mean(CSIIh);
NCMAvg = mean(NCM_val);
reset(ads_Tr);

zeroth_cell = 'A';
noise_cell = [zeroth_cell num2str(nn)];

first_cell = char(zeroth_cell + ss);

csii_cell = [first_cell num2str(nn)];
ncm_cell = [char(first_cell + 5) num2str(nn)];
stoi_cell = [char(first_cell + 10) num2str(nn)];

writematrix(noise_num(nn),filename,'Sheet',1,'Range',noise_cell);
writematrix(CSIIhAvg,filename,'Sheet',1,'Range',csii_cell);
writematrix(NCMAvg,filename,'Sheet',1,'Range',ncm_cell);
writematrix(STOIAvg,filename,'Sheet',1,'Range',stoi_cell);


writematrix(noise_num(nn),second_file,'Sheet',1,'Range',noise_cell);
writematrix(max(CSIIh),second_file,'Sheet',1,'Range',csii_cell);
writematrix(max(NCM_val),second_file,'Sheet',1,'Range',ncm_cell);
writematrix(max(STOI),second_file,'Sheet',1,'Range',stoi_cell);


end

end