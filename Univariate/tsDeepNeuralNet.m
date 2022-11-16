function [Output,MSE,W,b] = tsDeepNeuralNet (X,t,alpha,hiddenConfig,activation,normalize)
% ------------------------------------------------------------------------------------
% Deep Neural Network
% ------------------------------------------------------------------------------------
% INPUT: X: predictor variables
%        t: target variable
%        alpha: learning rate
%        hiddenConfig: hidden layers configuration, ex. [2;2;2] means that the network has 3
%        hidden layers, each of the layer has 2 neurons
%        activation: activation function (default: tanh)
%        normalize: select yes if the data not normalized (default: y)
% ------------------------------------------------------------------------------------
% OUTPUT: Output: predicted values
%	      MSE: mean square error
%	      W: weight matrix
%	      b: bias
% ------------------------------------------------------------------------------------
% Refrences:
% 1. M.T.Hagan, H.B.Demuth, M.H.Beale, O. De Jesús: Neural Network Design (2nd Edition) 
% 2. J.D.Kelleher, B.N.Namee, A. D'Arcy: Fundamentals of Machine Learning
% for Predictive Data Analytics (MIT Press, 2020)
% ------------------------------------------------------------------------------------
%
% Copyright: Laszlo Kamocsai
% https://github.com/lkamocsai
% lkamocsai@student.elte.hu
% Version: 1.0    Date: 11/10/2022
%
%-------------------------------------------------------------------------------------
%
%----------------------------------(1) check inputs ----------------------------------

arguments
    X {mustBeNonempty,mustBeNumeric}
    t {mustBeNonempty,mustBeNumeric}
    alpha {mustBeNonzero,mustBeNumeric} 
    hiddenConfig {mustBeNonempty,mustBeNonzero,mustBeNumeric}
    activation char {mustBeMember(activation,{'sigm','tanh','relu'})} = 'tanh'
    normalize char {mustBeMember(normalize,{'y','n'})} = 'y'
end

%----------------------------------(2) set activation --------------------------------

switch activation
    % Activation function and its first derivative
    case 'sigm'
        f_phi = @(x) 1./(1 + exp(-x)); 
        df_phi = @(x) (1 - x).*x;
    case 'tanh'
        f_phi = @(x) tanh(x);
        df_phi = @(x) 1-x.^2;
    case 'relu'
        f_phi = @(x) max(0,x);
        df_phi = @(x) x > 0;
end

%----------------------------------(2) normalize data --------------------------------

if strcmp(normalize,'y')
    data = [X t];
    minData = min(data,[],1);
    maxData = max(data,[],1);
    normData = bsxfun(@minus, data, minData);
    normData = bsxfun(@rdivide,normData,(maxData-minData));
    X = normData(:,1:end-1)';
    t = normData(:,end)';
else
    X = X';
    t = t';
end

%----------------------------------(3) config network --------------------------------

% Number of input variables
numIn = size(X,1); 
% Number of outputs
numOut = size(t,1); 
% Number of examples
numExamples = size(X,2); 
% Neurons per hidden layer, N x 1 vector
hiddenLayers = hiddenConfig;
% Network structure
nnStructure = [numIn;hiddenLayers(:);numOut]; 
% Network depth, layer 0 is the input
numLayers = numel(nnStructure) - 1; 

% Allocate memory for faster computing 
% Activations, a{1} = input data
a = cell(numLayers + 1,1); 
% Weighted sums
z = cell(numLayers,1); 
% Weights
W = cell(numLayers,1); 
% Bias
b = cell(numLayers,1); 
% Rate of change of the error respect to changes in weighet sum
delta = cell(numLayers,1); 
% Outputs
Output = zeros(1,numExamples);
Error = zeros(1,numExamples);

%rng(22)
% Generate random weights and bias
for i = 1:numLayers
    % generate (-0.5,0.5) random weights
    W{i} = rand(nnStructure(i + 1),nnStructure(i)) - 0.5;
    % generate (-0.5,0.5) random bias
    b{i} = rand(nnStructure(i + 1),1) - 0.5;
    % init delta
    delta{i} = zeros(nnStructure(i + 1),1);
end

%----------------------------------(4) train network ---------------------------------

for iter = 1:500000 % epocs
    % Start Stochastic Gradient Descent 
    for q = 1:numExamples    
        % qth example
        a{1} =X(:,q);
        % ------Start forward pass------
        for l = 1:numLayers
            % Weighted sum
            z{l} = (W{l} * a{l}) + b{l};
            % Activations
            if l < numLayers
                % Hidden layer
                a{l + 1} = f_phi(z{l});
            else
                % Output layer
                a{numLayers + 1} = (z{l});
            end
        end
        % ------End forward pass------
        
        % ------Start backpropagation------
        % the error gradient for single output neuron
        e(q) = (a{numLayers + 1} - t(q));
        
        % Output delta
        % Note, if the output layer activation is not linear use the following line: 
        % delta{numLayers} = df_phi(a{numLayers + 1})*e(q);
        delta{numLayers} = 1*e(q);

        % Hidden layers delta
        for l = numLayers-1:-1:1
            delta{l} = df_phi(a{l + 1}).* (W{l + 1}' * delta{l + 1});
        end
        % ----End backpropagation----
        
        % Update params
        for l = numLayers:-1:1
            dEdW{l} = (delta{l} * a{l}');
            dEdb{l} = (delta{l} * 1);
            W{l} = W{l} - alpha * dEdW{l};
            b{l} = b{l} - alpha * dEdb{l};
        end
        
        Output(iter,q) = a{numLayers + 1};
        Error(q) = 0.5 * ((a{numLayers + 1} - t(q)).^2);

    end
    % Mean square errors
    MSE(iter,:) = mean(Error);

    % Convergence criterion
    if MSE(iter,:) < 0.005
        break
    end
end

end