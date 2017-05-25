% competition III dataset IVa

% CSP with RLDA
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


% Regularization constant
lambda_list = logspace(-2,2,20);

for jj=1:length(subjects)

fprintf('Subject: %s\n', subjects{jj});

%% Load a dataset and preprocess
load(sprintf(file, subjects{jj}));

%% Select channels and covert cnt into double
cnt  = 0.1*double(cnt(:,opt.chanind));
clab = nfo.clab(opt.chanind);
C = length(clab);

%% Apply a band-pass filter

[b, a] = butter(opt.filtOrder, opt.band);
cnt_bp_a = filter(b,a,cnt); 

%% Cut EEG into tirals
xepo_a = cutoutTrials(cnt_bp_a, mrk.pos, opt.ival, nfo.fs);
X= covariance(xepo_a);
Y = (mrk.y-1.5)*2;  % convert {1,2} -> {-1, 1}

%% Find indices of training and test set
Itrain = find(~isnan(Y));
Itest  = find(isnan(Y));
Xtr = X(:,:,Itrain);
Ytr = Y(Itrain);
Y_tr = Ytr;
Y_tr(Y_tr~=1)=0;


%% Feature extraction

[Xtr, Ww] = whiten(Xtr);

if opt.logm
Xtr = logmatrix(Xtr);
end

%% Load the true label of the test set

load(sprintf(file_t, subjects{jj}));
Yte = (true_y(Itest)-1.5)*2;
Y_te = true_y(Itrain);
Y_te(Y_te~=1)=0;

Xte = X(:,:,Itest);

if opt.logm
Xte = logmatrix(matmultcv(Xte, Ww));
end

% Train and test data
xapp = reshape(Xtr,size(Xtr,1)*size(Xtr,2),size(Xtr,3));
xapp = xapp';

xtest = reshape(Xte,size(Xte,1)*size(Xte,2),size(Xte,3));
xtest = xtest';

%% Classification using LDA    
% validation parameters
%hold out/random
ratio=.6;

perf_type='ACC'

% train and test labels
yapp = Ytr';
ytest = Yte';

for ii=1:length(lambda_list);

lambda=lambda_list(ii);
estim=@(x,y) ldaclass(x,y,lambda);

% hold-out
loss_holdout(jj,ii)=valid_classif(xapp,yapp,estim,perf_type,'hold-out',ratio);

% model and true loss
[w,w0]=ldaclass(xapp,yapp,lambda);

ypred_app=xapp*w+w0;
ypred_test=xtest*w+w0;
perf_test=perf_classif(ytest,ypred_test);

true_loss(jj,ii)=perf_test.(perf_type);
end


%% LDA classifier + visu

[AUC,tpr,fpr,b]=svmroccurve(ypred_test,ytest);
AUC

end

for i = 1:length(subjects)
    Accuracy_train(i) = mean(loss_holdout(i,:));
end
%%
lw=2;
fs=16;

figure(1)
clf()
% set(gcf,'defaulttextinterpreter','latex');
plot(lambda_list,100*(loss_holdout),'linewidth',2)
hold on;
grid on;
plot(lambda_list, 100*(mean(loss_holdout)), 'color',[.7 .7 .7], 'linewidth', 2);
hold off
title('Motor Imagery Classification using LDA','FontSize',fs)
leg = [subjects {'average'}];
legend(leg);
% set(l,'FontSize',fs,'interpreter','latex')
xlabel(['lambda'],'FontSize',fs)
ylabel(['Classification Accuracy'],'FontSize',fs)

%% plot discriminant function

lw=2;
fonts=16;

figure(2)
set(gcf,'defaulttextinterpreter','latex');
plot(fpr,tpr)
hold on
plot([0 1],[0 1],'k')
hold off
title('Classification Motor Imagery','FontSize',fonts)

l=legend(['LDA, AUC=' num2str(AUC,3)]);
set(l,'FontSize',fonts,'interpreter','latex')
xlabel(['FPR'],'FontSize',fonts)
ylabel(['TPR'],'FontSize',fonts)

%% II Stochastic Gradient Boosting Algorithm

% SBCSP and boosting

Accu = [];

for jj=1:length(subjects)

fprintf('Subject: %s\n', subjects{jj});

%% Load a dataset and preprocess
load(sprintf(file, subjects{jj}));

%% Select channels and covert cnt into double
cnt  = 0.1*double(cnt(:,opt.chanind));
clab = nfo.clab(opt.chanind);
C = length(clab);

%% Apply a band-pass filter
% BCSSP
[cnt_1, cnt_2, cnt_3, cnt_4] =filterbank(cnt, nfo.fs);

s_a = max(cnt_1,cnt_2);
s_b = max(cnt_3,cnt_4);

cnt_bp = max(s_a, s_b);

%% Cut EEG into tirals
xepo = cutoutTrials(cnt_bp, mrk.pos, opt.ival, nfo.fs);

x_trl = xepo(:,:, mrk.y==1);
x_trr = xepo(:,:, mrk.y==2);
W = CSP(x_trl, x_trr);
W_csp  = [W(:,1) W(:,2)];
s = cnt_bp*W;
s_csp = cnt_bp * W_csp;

% spatial patterns of hand imagery is given by SP_H and SP_F for foot(csp  detail paper)
SP_H = W(1,:);
SP_F = W(2,:);

% variance
v1= var(W(1,:))
v2= var(W(2,:))

%the optimal channels determined through searching the maximums of the
%absolute value of SP_H and SP_F (C3 and CPz)
CH_H= max(abs(SP_H));
CH_F= max(abs(SP_F));

c1 = find(abs(SP_H) == CH_H);
c2 = find(abs(SP_F) == CH_F);

titl1 =cell2mat(nfo.clab(opt.chanind(1,c1)));
titl2 =cell2mat(nfo.clab(opt.chanind(1,c2)));

%% Topoplot
figure(3)
subplot(2,1,1)
Y_tr_H = find(Y_tr ==1);
Y_tr_F = find(Y_tr ==0);
Y_te_H = find(Y_te ==0);
Y_te_F = find(Y_te ==1);
% Plot 
[datavector_H, correlation_H] = topview(cnt_bp, W, C, Y_tr_H, Y_te_H, nfo.fs);
title('Hand Movement')
subplot(2,1,2)
[datavector_F, correlation_F] = topview(cnt_bp, W, C, Y_tr_F, Y_te_F, nfo.fs);
title('Foot Movement')

%% ERS/ERS plots
[erd_m1,ers_m1,t]=plot_erd(file,W,subjects, true_y, 9, 1);
[erd_m2,ers_m2,t]=plot_erd(file,W, subjects, true_y, 13, 2);

% [erd_b1,ers_b1,t]=plot_erd(file,W,subjects, true_y, c2, 1);
% [erd_b2,ers_b2,t]=plot_erd(file,W, subjects, true_y, c2, 2);

figure(4)
subplot(2,1,1)
plot(t, erd_m1, 'b', t, ers_m1, 'g')
ylabel([titl1, 'relative power [%]']);
title(['Electrode', titl1,'(Right Hand)']);
ylim([-150, 250]);
line([2, 2], [-150, 250], 'LineStyle', ':', 'Color', 'k');
xlabel('Time [s]');
legend('Mu', 'Beta');
grid('on');

subplot(2,1,2)
plot(t, erd_m2, 'b', t, ers_m2, 'g')
ylabel([titl2, 'relative power [%]']);
title(['Electrode', titl2,'(Right Foot)']);
ylim([-150, 250]);
line([2, 2], [-150, 250], 'LineStyle', ':', 'Color', 'k');
xlabel('Time [s]'); 
legend('Mu', 'Beta');
grid('on');

% subplot(2,2,3)
% plot(t, ers_m1, 'r', t, ers_m2, 'g')
% ylabel([titl1, 'relative power [%]']);
% title(['Electrode', titl1,'(Beta Band)']);
% ylim([-150, 150]);
% line([2, 2], [-150, 250], 'LineStyle', ':', 'Color', 'k');
% xlabel('Time [s]');
% legend('left', 'right');
% grid('on');
% 
% 
% subplot(2,2,4)
% plot(t, ers_b1, 'r', t, ers_b2, 'g')
% ylabel([titl2, 'relative power [%]']);
% title(['Electrode', titl2,'(Beta Band)']);
% ylim([-150, 150]);
% line([2, 2], [-150, 350], 'LineStyle', ':', 'Color', 'k');
% xlabel('Time [s]');
% legend('left', 'right');
% grid('on');

%% ERDS maps
h =[];
h.SampleRate = nfo.fs;
h.TRIG = mrk.pos';
h.Classlabel = true_y';

%  EEG FFT
    
[F, y1] = comp_fft(s,h, 1, 0, 3, 9);
[F, y2] = comp_fft(s,h, 2, 0, 3, 9);

figure(5)
plot(F, y1,'g', F, y2,'r'); 
xlim([0 125]) 
ylim([0, 1000]);
xlabel('Frequency (Hz)') 
ylabel('Amplitude') 
legend( 'right hand', 'right foot');

figure(6)
r1 = calcErdsMap(s(:,[9 13]) ,h, [0.0, 0.0, 3.5], [8, 30]);
plotErdsMap(r1);

%% Feature extraction
d = 9;
u = 0.0085;
bands = [8,14;19,24;24,30];
win = 2;
[ff, gg] = tdp(s_csp, d, u);

%'log-power+log-amplitude')
f = [ff,gg];

%bandpower
% f = bandpower(s_csp, nfo.fs, bands, win);

% parameters
t = [];
t.low = 0.5;
t.high = 2.5;
gap = 0;
pre = ceil(t.low * nfo.fs);
post = ceil(t.high * nfo.fs);
opt.frame = [pre post];
[train_feats,sz] = trigg(f, mrk.pos', pre, post, gap);
train_feats = reshape(train_feats, sz);
train_feats = reshape(train_feats ,size(train_feats ,1)*size(train_feats ,2),size(train_feats ,3));

% Feature selection using fisher score
[out] = fisherscore(train_feats,true_y);
z = find(out.W > mean(out.W));
z  = sort(z, 'ascend');
train_feats = train_feats(:,z);
label = true_y;
label = label(:,z);

% Boosting

fprintf('Stochastic Gradient Boosting\n');
label(label~=1)=0;

I_test = find(isnan(mrk.y));
test_sz = size(I_test,2);
[correct, Accuracy]= stochgradboost(train_feats, label, C);
fprintf('Accuracy=%f\n', 100*Accuracy)
Accu = [Accu Accuracy];
 end

% Plot
figure(8)
data = [100*(Accuracy_train(:)), 100*(Accu(:))];
hb = bar(data);
set(hb(1), 'FaceColor','r')
set(hb(2), 'FaceColor','b')
ylim([50, 100])
xlabel('Subjects')
ylabel('Classification Accuracy')
set(gca,'xticklabel', {'aa', 'al', 'av', 'aw', 'ay'})
legend('RCSP', 'Boosting based CSSP','Location','SouthEast')
title('Classification accuracy for BCI competition III IVa dataset')

