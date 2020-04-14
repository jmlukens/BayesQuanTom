function timeTestPL
% This function estimates the computational cost (in terms of time) for
% sampling from a pseudo-likelihood, using data from simulatedData.m.

% Joseph M. Lukens (lukensjm@ornl.gov)
% 2020.04.14
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ %
clear all;  close all;

%% INPUTS
numSamp = 2^14;             % Number of samples to obtain from MH.
fileNum = '001';            % Number used in saved file.
dataFileName = 'simData_L=0.95_d=7.mat';          % Input data.
d = 7;                      % Single-qudit dimension.
D = d^2;                    % Two-qudit dimension.
BURN = 2^10;                % Burn-in period.

numParam = 2*D^2+D;         % Number of parameters to find.
alpha = 1;                  % Parameters for gamma distribution prior.

beta1 = 0.1;            % Initial parameters for stepsize.
beta2 = 0.1;

% Load data from simulated experiment:
load(dataFileName,'psi0','counts','rhoLS')
PHI = psi0;                 % Ideal state.
N = sum(counts);            % Total number of events.
sigma = 1/sqrt(N);          % Standard deviation of pseudo-likelihood.
rhoLSvec = reshape(rhoLS.',[],1);     % Convert to column vector.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INITIAL VALUES FOR SAMPLER
rng('shuffle')
param0 = zeros(1,numParam);
param0(:,1:2*D^2) = randn(1,2*D^2);
param0(:,2*D^2+1:end) = randg(alpha,1,D).';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MH SAMPLER
tic;
sampTemp = zeros(numSamp+BURN,numParam);    % Initializing samples.
costTemp = zeros(numSamp+BURN,1);           % Keeping track of time taken per iteration.
acc = 0;                            % Counter of acceptances.

% Initial point:
x = param0;
rhoX = paramToRhoCol(x);
logX = -1/(2*sigma^2)*norm(rhoX-rhoLSvec)^2 ...
    + sum(alpha*log(x(2*D^2+1:end)) - x(2*D^2+1:end));      % PL & correction factor.

for j=1:numSamp+BURN
    a = tic;
    % Proposed updated parameters:
    newGauss = sqrt(1-beta1^2)*x(1:2*D^2) + beta1*randn(1,2*D^2);
    newGamma = x(2*D^2+1:end).*exp(beta2*randn(1,D));
    y = [newGauss newGamma];
    
    rhoY = paramToRhoCol(y);
    logY = -1/(2*sigma^2)*norm(rhoY-rhoLSvec)^2 ...
        + sum(alpha*log(y(2*D^2+1:end))-y(2*D^2+1:end));
    
    if log(rand) < logY - logX
        x = y;      % Accept new point.
        logX = logY;
        acc = acc+1;
    end
    
    sampTemp(j,:) = x;     % Store samples.
    costTemp(j) = toc(a);
end

samp = sampTemp(BURN+1:end,:);      % Remove burn-in region.
cost = costTemp(BURN+1:end);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% WRITING TO FILE
Today = date;

FileName = ['timeTestPLdata_' datestr(Today,'yyyy') datestr(Today,'mm') ...
    datestr(Today,'dd') '_' fileNum];
save(FileName,'samp','cost','alpha','dataFileName','d')

fprintf(['Cost = ' num2str(mean(cost)) ' +/- ' num2str(std(cost)) ' sec\n'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SUB-ROUTINES
% Converts parameter set (row vector) into density matrix (expressed as a column vector).
    function z = paramToRhoCol(par)
        Xr = reshape(par(1:D^2),[D D]);
        Xi = reshape(par(D^2+1:2*D^2),[D D]);
        X = Xr + i*Xi;                      % Matrix of column vectors (unnormalized).
        
        NORM = vecnorm(X,2,1);              % Compute the norm of each column.
        W = X./NORM;                        % Normalize each column.
       
        Y = par(2*D^2+1:end);                     % Projector weights.        
        gamma = Y/sum(Y);                   % Normalize.
        
        rho = W*diag(gamma)*W';             % Density matrix.
        z = reshape(rho.',[],1);            % Convert to column, in order of rho(1,1), rho(1,2), rho(1,3),...,etc.
    end

end
