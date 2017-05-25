function [s_avg_erdm,s_avg_erdb,t] = plot_erd(file, W, subjects, y, channel, class)

jj = 1;
load(sprintf(file, subjects{jj}));
cnt_d = 0.1*double(cnt);
s = cnt_d;
eegchan =[1:21];
s = s(:, eegchan);
s = s*W;

fs = nfo.fs;
erdm_bandpass = [8,12];
erdb_bandpass = [14, 18];
%b_erd = fir1(fs, erd_bandpass/(fs/2));
[b_erdm, a_erdm] = butter(5, erdm_bandpass/(fs/2));
[b_erdb, a_erdb] = butter(5, erdb_bandpass/(fs/2));

st = trigg(s, mrk.pos(ismember(y, class)), round(0*fs), round(3.5*fs));
s2d = reshape(st(channel,:),[390, 126]);

s_filt = filter(b_erdm, a_erdm, s2d);
s_sq = s_filt.^2;
s_avg_erdm = sum(s_sq, 2) / size(s2d,2);

baseline = mean(s_avg_erdm(1*fs:2*fs));
s_avg_erdm = (s_avg_erdm - baseline) / baseline * 100;


s_filt_erdb = filter(b_erdb, a_erdb, s2d);
s_sq_erdb = s_filt_erdb.^2;
s_avg_erdb = sum(s_sq_erdb, 2) / size(s2d,2);

baseline = mean(s_avg_erdb(1*fs:2*fs));
s_avg_erdb = (s_avg_erdb - baseline) / baseline * 100;
t = (0:size(s_avg_erdm)-1)/fs;

%LOWPASS FILTERING
[b, a] = butter(5, 10./(fs/2));
s_avg_erdm = filter(b,a,s_avg_erdm)';
s_avg_erdb = filter(b,a,s_avg_erdb)';
t = t';

end
