
function [datavector, correlation] = topview(s,w,num_channels,y,v,fs)
%% Draw topoplot
s_csp = s*w;
y=y';
flashing = y(2:end);
flashing2 = y(1:end-1);
changes =[0; flashing - flashing2];

index = find(changes); %location of sdifferent intensifications 
% index =[1 index];
num_samples = size(index,1);
data = zeros(num_samples, num_channels, length(index));

    for j = 1: length(index)
        data (:,:,j) = s_csp(index(j):index(j) + num_samples-1, :);
    end
    
% Compute the correlation between the StimulusType and the response
% amplitude for each time sample and channel.
data = permute(data,[3 1 2]); 

correlation = zeros(num_channels, num_samples);
v=v';
for i = 1:num_channels
    for j = 1:num_samples
        correlation(i, j) = corr(squeeze(data(:, j, i)), double(v(index)));
    end
end

datavector = correlation(:, (0.1 * fs)-1 );
topoplot(datavector,'eloc21.txt','EEG');

end
