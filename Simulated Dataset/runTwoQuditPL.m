 function runTwoQuditPL
% This function performs Bayesian state estimation for simulated two-qudit
% experiments, for many values of THIN to gauge convergence. It uses (i)
% pCN sampling and (ii) the pseudo-likelihood.

% Joseph M. Lukens (lukensjm@ornl.gov)
% 2020.04.14
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ %
clear all;  close all;
t=tic;
%% LOOP PARAMETERS
THIN = 2.^(0:1:7);       % THIN values considered for each sampler.
samplers = 1;         % Number of independent samplers per THIN value.
fileNum = '001';        % Number used in saved file.
dataFileName = 'simData_L=0.85_d=3.mat';          % Input data.

%% INPUTS
numSamp = 2^10;             % Number of samples to obtain from pCN.
alpha = 1;                  % Parameter for gamma distribution prior.

Mb = 500;           % Update stepsize after this many points.
r = 1.1;            % Acceptance rate update factor.

% Load data from simulated experiment:
load(dataFileName,'psi0','counts','rhoLS')
PHI = psi0;     % Ideal state.
d = sqrt(length(psi0));     % Single-qudit dimension.
D = d^2;                    % Two-qudit dimension.
numParam =  2*D^2+D;        % Number of parameters to find.

N = sum(counts);            % Total number of events.
sigma = 1/sqrt(N);          % Standard deviation of pseudo-likelihood.
rhoLSvec = reshape(rhoLS.',[],1);     % Convert to column vector.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SAMPLING LOOP
Fmean = zeros(samplers,length(THIN));       % Initializing results to save.
Fstd = zeros(samplers,length(THIN));
samplerTime = zeros(samplers,length(THIN));    

handle = @paramToRhoCol;            % Needed if one wishes to use parallel for-loop.

for k=1:length(THIN)
    for m=1:samplers
        param0 = zeros(1,numParam);
        Fest=zeros(numSamp,1);
        
        rng('shuffle')
        param0(1:2*D^2) = randn(1,2*D^2);     % Initial seed.
        param0(2*D^2+1:end) = gamrnd(alpha,1,1,D);
        
        beta1 = 0.1;        % Initial parameters for stepsize.
        beta2 = 0.1;
        acc = 0;                            % Counter of acceptances.
        
        % Initial point:
        x = param0;
        rhoX = feval(handle,x);
        logX =  -1/(2*sigma^2)*norm(rhoX-rhoLSvec)^2 ...
            + sum(alpha*log(x(2*D^2+1:end)) - x(2*D^2+1:end));      % PL & correction factor.

        % pCN Loop
        tic
        for j=1:numSamp*THIN(k)
            % Proposed updated parameters:
            newGauss = sqrt(1-beta1^2)*x(1:2*D^2) + beta1*randn(1,2*D^2);
            newGamma = x(2*D^2+1:end).*exp(beta2*randn(1,D));
            y = [newGauss newGamma];
            
            rhoY = feval(handle,y);
            logY =  -1/(2*sigma^2)*norm(rhoY-rhoLSvec)^2 ...
                + sum(alpha*log(y(2*D^2+1:end))-y(2*D^2+1:end));
            
            if log(rand) < logY - logX
                x = y;      % Accept new point.
                logX = logY;
                acc = acc+1;
            end
            
            if mod(j,Mb)==0             % Stepsize adaptation.
                rat = acc/Mb;           % Estimate acceptance probability, and keep near 0.234.
                if rat>0.3
                    beta1=beta1*r;
                    beta2=beta2*r;
                elseif rat<0.1
                    beta1=beta1/r;
                    beta2=beta2/r;
                end
                acc=0;
            end
            
            if mod(j,THIN(k)) == 0      % Keep sample & compute F.
                rhoAsVec = feval(handle,x);
                rhoEst = reshape(rhoAsVec,[D D]).'; % Watch map convention here.
                Fest(j/THIN(k)) = real(PHI'*rhoEst*PHI);
            end
        end
        samplerTime(m,k) = toc;
        
        % Save quantities of interest.
        Fmean(m,k) = mean(Fest);        % Mean & STD of fidelity for particular sample run.
        Fstd(m,k) = std(Fest);
        
        fprintf([num2str((k-1)*samplers+m) ' of ' ...
            num2str(length(THIN)*samplers) ' (' ...
            num2str(samplerTime(m,k)) ' s) F=' ...
            num2str(Fmean(m,k)) ' +/- ' num2str(Fstd(m,k)) '\n'])
    end
end


%% WRITING TO FILE
Today = date;

FileName = ['runTwoQuditPLdata_' datestr(Today,'yyyy') datestr(Today,'mm') ...
    datestr(Today,'dd') '_' fileNum];
save(FileName,'THIN','samplers','samplerTime','Fmean','Fstd','alpha','dataFileName')

%% PLOTTING OUTPUT
log2THIN = repmat(log2(THIN),samplers,1);

figure
subplot(1,3,1)
hold on;    box on;
scatter(log2THIN(:),Fmean(:),50,'filled')
axis([0 max(log2(THIN)) 0 1])
set(gca,'FontName','Arial','FontSize',16)
xlabel('log_2(THIN)')
ylabel('\langleF\rangle')

subplot(1,3,2)
hold on;    box on;
scatter(log2THIN(:),Fstd(:),50,'filled')
axis([0 max(log2(THIN)) 0 Inf])
set(gca,'FontName','Arial','FontSize',16)
xlabel('log_2(THIN)')
ylabel('\DeltaF')
title(['\alpha = ' num2str(alpha)])


subplot(1,3,3)
hold on;    box on;
scatter(log2THIN(:),samplerTime(:),50,'filled')
axis([0 max(log2(THIN)) 0 Inf])
set(gca,'FontName','Arial','FontSize',16)
xlabel('log_2(THIN)')
ylabel('Runtime [s]')


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
toc(t)
end