%% BCI competition III IVA dataset, Binary classification using LDA with a regularization constant(lambda)

clc;
clear all;
close all;

file   = 'data_set_IVa_%s.mat';
file_t = 'true_labels_%s.mat';

subjects = {'aa','al','av','aw','ay'};

opt.ival= [500 2500];
opt.filtOrder= 5;
opt.band = [8/50 30/50];
opt.logm = 0;

%% Reduced set of 21 channels
opt.chanind = [33, 34, 35, 36, 37, 38, 39, 51, 52, 53, 54, 55, 56, 57, ...
   69, 70, 71, 72, 73, 74, 75];


Accuracy_train = [];
Accuracy_test = [];

for jj=1:length(subjects)

fprintf('Subject: %s\n', subjects{jj});

%% Load a dataset and preprocess
load(sprintf(file, subjects{jj}));

%% Select channels and covert cnt into double
cnt  = 0.1*double(cnt(:,opt.chanind));
clab = nfo.clab(opt.chanind);
C = length(clab);

%% Apply a band-pass filter

[cnt_1, cnt_2, cnt_3, cnt_4] =filterbank(cnt, nfo.fs);

s_a = max(cnt_1,cnt_2);
s_b = max(cnt_3,cnt_4);

cnt_bp = max(s_a, s_b);

%% Cut EEG into tirals
xepo = cutoutTrials(cnt_bp, mrk.pos, opt.ival, nfo.fs);

X= covariance(xepo);
Y = (mrk.y-1.5)*2;  % convert {1,2} -> {-1, 1}

%% Find indices of training and test set
Itrain = find(~isnan(Y));
Itest  = find(isnan(Y));

%% Whiten the training data
Xtr = X(:,:,Itrain);
Ytr = Y(Itrain);

[Xtr, Ww] = whiten(Xtr);

if opt.logm
Xtr = logmatrix(Xtr);
end

%% Load the true label of the test set

load(sprintf(file_t, subjects{jj}));
Yte = (true_y(Itest)-1.5)*2;

Xte = X(:,:,Itest);

if opt.logm
Xte = logmatrix(matmultcv(Xte, Ww));
end

%% Topoplot
Y_tr = Ytr;
Y_tr(Y_tr~=1)=0;
Y_te = true_y(Itrain);
Y_te(Y_te~=1)=0;

figure(1)
[datavector, correlation] = topview(cnt_bp, Ww, C, Y_tr, Y_te, nfo.fs);


%% ERS/ERS plots
[erda_m1,erdb_m1,t]=plot_erd(file, Ww, subjects, true_y, 9, 1);
[erda_m2,erdb_m2,t]=plot_erd(file, Ww, subjects, true_y, 9, 2);

[erda_b1,erdb_b1,t]=plot_erd(file, Ww, subjects, true_y, 13, 1);
[erda_b2,erdb_b2,t]=plot_erd(file, Ww, subjects, true_y, 13, 2);

figure(2)
subplot(2,2,1)
plot(t, erda_m1, 'r', t, erda_m2, 'g')
ylabel('C3 relative power [%]');
title('Electrode C3 (Mu Band)');
ylim([-150, 400]);
line([2, 2], [-150, 250], 'LineStyle', ':', 'Color', 'k');
xlabel('Time [s]');
legend('left', 'right');
grid('on');

subplot(2,2,2)
plot(t, erda_b1, 'r', t, erda_b2, 'g')
ylabel('C4 relative power [%]');
title('Electrode C4 (Mu Band)');
ylim([-150, 400]);
line([2, 2], [-150, 250], 'LineStyle', ':', 'Color', 'k');
xlabel('Time [s]'); 
legend('left', 'right');
grid('on');

subplot(2,2,3)
plot(t, erdb_m1, 'r', t, erdb_m2, 'g')
ylabel('C3 relative power [%]');
title('Electrode C3 (Beta Band)');
ylim([-150, 400]);
line([2, 2], [-150, 350], 'LineStyle', ':', 'Color', 'k');
xlabel('Time [s]');
legend('left', 'right');
grid('on');


subplot(2,2,4)
plot(t, erdb_b1, 'r', t, erdb_b2, 'g')
ylabel('C4 relative power [%]');
title('Electrode C4 (Beta Band)');
ylim([-150, 400]);
line([2, 2], [-150, 350], 'LineStyle', ':', 'Color', 'k');
xlabel('Time [s]');
legend('left', 'right');
grid('on');

%% Classification using LDA 

% train and test features
xapp = reshape(Xtr,size(Xtr,1)*size(Xtr,2),size(Xtr,3));
xapp = xapp';
xtest = reshape(Xte,size(Xte,1)*size(Xte,2),size(Xte,3));
xtest = xtest';
% train and test labels
yapp = Ytr';
ytest = Yte';

%% LDA classifier + visu

lambda=1e2;
[w,w0]=ldaclass(xapp,yapp,lambda)

ypred_app=xapp*w+w0;
ypred_test=xtest*w+w0;

ACC_app=mean(sign(ypred_app)==yapp)
ACC_test=mean(sign(ypred_test)==ytest)

Accuracy_train = [Accuracy_train  ACC_app];
Accuracy_test = [Accuracy_test  ACC_test];

[AUC,tpr,fpr,b]=svmroccurve(ypred_test,ytest);
AUC

%% %% plot Accuracy for 5 subjects

figure(3)
bar(Accuracy_train*100);xlabel('Subjects');ylabel('Accuracy(%)')

end

%% plot discriminant function

lw=2;
fs=14;

figure(4)
set(gcf,'defaulttextinterpreter','latex');
plot(fpr,tpr)
hold on
plot([0 1],[0 1],'k')
title('Classification of Motor Imagery','FontSize',fs)

l=legend(['LDA, AUC=' num2str(AUC,3)]);
set(l,'FontSize',fs,'interpreter','latex')
xlabel(['FPR'],'FontSize',fs)
ylabel(['TPR'],'FontSize',fs)
