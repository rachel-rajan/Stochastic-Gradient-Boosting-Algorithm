function [f, y_fft] = comp_fft(s,h, class, pre, post, channel)
% pre=0, post=7.5, channel=12
fs = h.SampleRate;
%class = 1;
gap = 0;

st = trigg(s, h.TRIG(ismember(h.Classlabel, class)), round(pre*fs), round(post*fs));

st = st(channel, :);
st = reshape(st, [size(st,2)/86, 86]);

Nsamps = length(st); 

y_fft = abs(fft(st));
y_fft = mean(y_fft, 2); 
y_fft = y_fft(1:Nsamps/2);
f = fs*(0:Nsamps/2-1)/Nsamps;



end

