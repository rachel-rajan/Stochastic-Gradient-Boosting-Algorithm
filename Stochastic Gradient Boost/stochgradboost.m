function [correct, Accuracy]= stochgradboost(f, y, n_channels)

% reshape f
% f = reshape(f,size(f,1)*size(f,2),size(f,3));
x = f;

% prepare index sets for cross-validation
n_permutations = 5;
n_epochs = size(x,2);
testsetsize = round(n_epochs / 10);
[trainsets, testsets] = crossValidation(1:n_epochs, testsetsize, ...
                                       n_permutations);
% cross-validation loop
correct = [];
for i = 1:n_permutations
    
    % draw data from CV index sets
    train_x = x(:, trainsets(i,:));
    train_y = y(:, trainsets(i,:));
    test_x  = x(:, testsets(i,:));
    test_y  = y(:, testsets(i,:));    
    
    % train classifier and apply to test data
    l = LogitBoost(180,0.05, 1);
    l = train(l, train_x, train_y, n_channels);
    p = classify(l, test_x);    

    % evaluate classification accuracy 
    i0 = find(p <= 0.5);
    i1 = find(p > 0.5);
    est_y = zeros(size(p));
    est_y(i0) = 0;
    est_y(i1) = 1;
    for j = 1:size(est_y,1)
        n_correct(j) = length(find(est_y(j,:) == test_y));
    end
    p_correct = n_correct / size(est_y,2);
    correct = [correct ; p_correct];    
    
    % plot number of steps vs. classification accuracy 
    if (i>1)
        plot(mean(correct));
        xlabel('number of boosting iterations');
        ylabel('classification accuracy');
        drawnow;
    end
    
    Accuracy = max(mean(correct));
    
 end