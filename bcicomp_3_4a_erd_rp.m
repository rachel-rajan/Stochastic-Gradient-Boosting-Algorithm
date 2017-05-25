% competition III dataset IVa

% CSP with RLDA
clc;
clear all;
close all;

file   = 'data_set_IVa_%s.mat';
file_t = 'true_labels_%s.mat';

subjects = {'aa','al','av','aw','ay'};

opt.ival_erd = [500 2500];
opt.ival_rp = [300 3500];
opt.filtOrder= 5;
opt.band_erd = [8/50 30/50];
opt.band_rp = [2.5/50 3.5/50];
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
[b_erd, a_erd] = butter(opt.filtOrder, opt.band_erd);
[b_rp, a_rp] = butter(opt.filtOrder, opt.band_rp);
cnt_bp_erd = filter(b_erd,a_erd,cnt); 
cnt_bp_rp = filter(b_rp,a_rp,cnt); 


%% Cut EEG into tirals
xepo_erd = cutoutTrials(cnt_bp_erd, mrk.pos, opt.ival_erd, nfo.fs);
xepo_rp = cutoutTrials(cnt_bp_rp, mrk.pos, opt.ival_rp, nfo.fs);

% Baseline correction for RP
Y_c = permute(xepo_rp,[2,1,3]);
baseline_end_frame = floor(size(Y_c,2)/2048*300);
baseline = mean(Y_c(:,1:baseline_end_frame),2);
Y_corrected = bsxfun(@minus,Y_c,baseline);
xepo_rp = permute(Y_corrected,[2,1,3]);

X_erd= covariance(xepo_erd);
X_rp = covariance(xepo_rp);

Y = (mrk.y-1.5)*2;  % convert {1,2} -> {-1, 1}

%% Find indices of training and test set
Itrain = find(~isnan(Y));
Itest  = find(isnan(Y));
Xtr_erd = X_erd(:,:,Itrain);
Xtr_rp = X_rp(:,:,Itrain);
Ytr = Y(Itrain);
Y_tr = Ytr;
Y_tr(Y_tr~=1)=0;

%% Feature extraction

[Xtr_erd, Ww_erd] = whiten(Xtr_erd);

if opt.logm
Xtr_erd = logmatrix(Xtr_erd);
end

[Xtr_rp, Ww_rp] = whiten(Xtr_rp);

if opt.logm
Xtr_rp = logmatrix(Xtr_rp);
end

%% Load the true label of the test set

load(sprintf(file_t, subjects{jj}));
Yte = (true_y(Itest)-1.5)*2;
Y_te = true_y(Itrain);
Y_te(Y_te~=1)=0;

Xte = [X_erd(:,:,Itest), X_rp(:,:,Itest)];
Ww = [Ww_erd, Ww_rp];

if opt.logm
Xte = logmatrix(matmultcv(Xte, Ww));
end

% Train and test data
xapp_erd = reshape(Xtr_erd,size(Xtr_erd,1)*size(Xtr_erd,2),size(Xtr_erd,3));
xapp_rp = reshape(Xtr_rp,size(Xtr_rp,1)*size(Xtr_rp,2),size(Xtr_rp,3));
xapp = [xapp_erd' xapp_rp'];

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


%% II Stochastic Gradient Boosting 

% SBCSP + boostings

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

cnt_bp_erds = max(s_a, s_b);

[b, a] = butter(opt.filtOrder, opt.band_rp);
cnt_bp_rdp = filter(b, a, cnt); 

%% Cut EEG into tirals
xepo_erds = cutoutTrials(cnt_bp_erds, mrk.pos, opt.ival_erd, nfo.fs);
xepo_rdp = cutoutTrials(cnt_bp_rdp, mrk.pos, opt.ival_rp, nfo.fs);

% Baseline correction for RP
Y_c = permute(xepo_rdp,[2,1,3]);
baseline_end_frame = floor(size(Y_c,2)/2048*300);
baseline = mean(Y_c(:,1:baseline_end_frame),2);
Y_corrected = bsxfun(@minus,Y_c,baseline);
xepo_rdp = permute(Y_corrected,[2,1,3]);

% 1. ERD
x_trl = xepo_erds(:,:, mrk.y==1);
x_trr = xepo_erds(:,:, mrk.y==2);
W_erds = CSP(x_trl, x_trr);
W_csp_erds  = [W_erds(:,1) W_erds(:,2)];
s_csp_erds = cnt_bp_erds * W_csp_erds;

% 2. RP
x_trl_rdp = xepo_rdp(:,:, mrk.y==1);
x_trr_rdp = xepo_rdp(:,:, mrk.y==2);
W_rdp = CSP(x_trl_rdp, x_trr_rdp);
W_csp_rdp  = [W_rdp(:,1) W_rdp(:,2)];
s_csp_rdp = cnt_bp_rdp * W_csp_rdp;

% spatial patterns of hand imagery is given by SP_H and SP_F for foot(csp  detail paper)
SP_H = W_erds(1,:);
SP_F = W_erds(2,:);

% variance
v1= var(W_erds(1,:))
v2= var(W_erds(2,:))

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
[datavector_H, correlation_H] = topview(cnt_bp_erds, W_erds, C, Y_tr_H, Y_te_H, nfo.fs);
title('Hand Movement')
subplot(2,1,2)
[datavector_F, correlation_F] = topview(cnt_bp_erds, W_erds, C, Y_tr_F, Y_te_F, nfo.fs);
title('Foot Movement')

%% ERS/ERS plots
[erd_m1,ers_m1,t]=plot_erd(file,W,subjects, true_y, c1, 1);
[erd_m2,ers_m2,t]=plot_erd(file,W, subjects, true_y, c2, 2);

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

%% Feature extraction
d = 9;
u = 0.0085;
bands = [8,14;19,24;24,30];
win = 2;
[ff, gg] = tdp(s_csp_erds, d, u);
[ff_rdp, gg_rdp] = tdp(s_csp_rdp, d, u);

%'log-power+log-amplitude')
f = [ff,gg,ff_rdp,gg_rdp];

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
figure(5)
fprintf('Stochastic Gradient Boosting\n');

label(label~=1)=0;

I_test = find(isnan(mrk.y));
test_sz = size(I_test,2);
[correct, Accuracy]= stochgradboost(train_feats, label, C);
fprintf('Accuracy=%f\n', 100*Accuracy)
Accu = [Accu Accuracy];
end

% Plot
data = [100*(Accuracy_train(:)), 100*(Accu(:))];
hb = bar(data);
set(hb(1), 'FaceColor','r')
set(hb(2), 'FaceColor','b')
ylim([50, 100])
ylabel('Classification Accuracy')
set(gca,'xticklabel', {'aa', 'al', 'av', 'aw', 'ay'})
legend('RCSP', 'Boosting based CSSP', 'Location', 'SouthEast')
title('Classification accuracy  for BCI competition III IVa (Feature Combination)')

