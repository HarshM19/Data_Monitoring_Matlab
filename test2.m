clear all;
clc;
%%  divide dataset
x=ExampleData(:,1:4);
y=ExampleData(:,5);

rand=randperm(100);

xtr=x(rand(1:80),:);
ytr=y(rand(1:80),:);

xt=x(rand(81:end),:);
yt=y(rand(81:end),:);

%% Train
model = fitcsvm(xtr ,ytr ,'KernelFunction', 'rbf' ,...
    'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus','ShowPlots', true));
%% Testing Model
result = predict(model,xt);
accuracy = sum(result == yt)/length(yt)*100;
sp= sprintf("Test Accuracy = %.2f",accuracy);
disp(sp);
