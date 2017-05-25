% Filterbank implementaion SBCSP

function [data_1, data_2, data_3, data_4] = filterbank(data, fs)

% data --> EEG data(channels x samples )
% fs --> Sampling frequeny

band1 = [8 13]/(fs/2);
%[b a] = butter(n, Wn)
[b1, a1] = butter(5, band1);
data_1 = filtfilt(b1, a1, data); % Band pass filtering

band2 = [13 18]/(fs/2);
[b2, a2] = butter(5, band2);
data_2 = filtfilt(b2, a2, data);

band3 = [18 25]/(fs/2);
[b3, a3] = butter(5, band3);
data_3 = filtfilt(b3, a3, data);

band4 = [25 30]/(fs/2);
[b4, a4] = butter(5, band4);
data_4 = filtfilt(b4, a4, data);
end