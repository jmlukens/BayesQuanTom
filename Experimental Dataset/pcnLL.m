 function pcnLL
% This function is designed to run many samplers performing Bayesian
% quantum state tomography on the experimental data from [Optica 5, 1455
% (2018)].

% It utilizes: (i) pCN Metropolis-Hastings and (ii) a full multinomial
% likelihood.

% Joseph M. Lukens (lukensjm@ornl.gov)
% 2020.04.14
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ %
clear all;  close all;
%% LOOP PARAMETERS
THIN = 2.^[0 1 2 3 4 5 6 7];       % THIN values considered for each sampler.
samplers = 5;           % Number of independent samplers per THIN value.
fileNum = '001';        % Number used in saved file.

%% INPUTS
numSamp = 2^10;         % Number of samples to obtain from pCN.

PHI = 1/sqrt(2)*[0 1 1 0]';     % Ideal state.
alpha = 1;          % Parameter for gamma distribution prior.

Mb = 500;           % Update stepsize after this many points.
r = 1.1;            % Acceptance rate update factor.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MAP FROM DENSITY MATRIX TO PROBABILITIES
% 16x16 Transformation from density matrix (as vector) to probabilities of outcomes:

%                                     INDEX MAP (m,n)
%     11   12   13   14   21   22   23   24   31   32   33   34   41   42   43   44        MEASUREMENT OUTCOMES
U = [  1    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0   ; ...     % ZZ 00
       0    0    0    0    0    1    0    0    0    0    0    0    0    0    0    0   ; ...     % ZZ 01
       0    0    0    0    0    0    0    0    0    0    1    0    0    0    0    0   ; ...     % ZZ 10
       0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    1   ; ...     % ZZ 11
      1/2  1/2   0    0   1/2  1/2   0    0    0    0    0    0    0    0    0    0   ; ...     % ZX 00
      1/2 -1/2   0    0  -1/2  1/2   0    0    0    0    0    0    0    0    0    0   ; ...     % ZX 01
       0    0    0    0    0    0    0    0    0    0   1/2  1/2   0    0   1/2  1/2  ; ...     % ZX 10
       0    0    0    0    0    0    0    0    0    0   1/2 -1/2   0    0  -1/2  1/2  ; ...     % ZX 11
      1/2   0   1/2   0    0    0    0    0   1/2   0   1/2   0    0    0    0    0   ; ...     % XZ 00
       0    0    0    0    0   1/2   0   1/2   0    0    0    0    0   1/2   0   1/2  ; ...     % XZ 01   
      1/2   0  -1/2   0    0    0    0    0  -1/2   0   1/2   0    0    0    0    0   ; ...     % XZ 10
       0    0    0    0    0   1/2   0  -1/2   0    0    0    0    0  -1/2   0   1/2  ; ...     % XZ 11
      1/4  1/4  1/4  1/4  1/4  1/4  1/4  1/4  1/4  1/4  1/4  1/4  1/4  1/4  1/4  1/4  ; ...     % XX 00 
      1/4 -1/4  1/4 -1/4 -1/4  1/4 -1/4  1/4  1/4 -1/4  1/4 -1/4 -1/4  1/4 -1/4  1/4  ; ...     % XX 01
      1/4  1/4 -1/4 -1/4  1/4  1/4 -1/4 -1/4 -1/4 -1/4  1/4  1/4 -1/4 -1/4  1/4  1/4  ; ...     % XX 10
      1/4 -1/4 -1/4  1/4 -1/4  1/4  1/4 -1/4 -1/4  1/4  1/4 -1/4  1/4 -1/4 -1/4  1/4 ];         % XX 11


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DATA
% Counts for all basis combinations (coincidences only):
counts = [     7    304    280      8 ...       % ZZ
             151    128    154    159 ...       % ZX
             143    147    135    159 ...       % XZ
             289     18     12    297 ];        % XX

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SAMPLING LOOP
Fmean = zeros(samplers,length(THIN));       % Initializing results to save.
Fstd = zeros(samplers,length(THIN));
samplerTime = zeros(samplers,length(THIN));         
         
samp = zeros(numSamp,36);     % Allocating variable space.
param0 = zeros(1,36);

for k=1:length(THIN)
    for m=1:samplers
        
        rng('shuffle')
        param0(1:32) = randn(1,32);     % Initial seed.
        param0(33:36) = randg(alpha,1,4);
        
        Fest = zeros(numSamp,1);            % Fidelities estimated for each sample.
        
        beta1 = 0.1;        % Initial parameters for stepsize.
        beta2 = 0.1;
        acc = 0;                            % Counter of acceptances.
        
        % Initial point:
        x = param0;
        rhoX = paramToRhoCol(x);
        logX = counts*log(U*rhoX) + sum(alpha*log(x(33:36)) - x(33:36));        % Likelihood & correction factor.
        
        % pCN Loop
        tic
        for j=1:numSamp*THIN(k)
            % Proposed updated parameters:
            newGauss = sqrt(1-beta1^2)*x(1:32) + beta1*randn(1,32);
            newGamma = x(33:36).*exp(beta2*randn(1,4));
            y = [newGauss newGamma];
            
            rhoY = paramToRhoCol(y);
            logY = counts*log(U*rhoY) + sum(alpha*log(y(33:36)) - y(33:36));
            
            if log(rand) < logY - logX
                x = y;      % Accept new point.
                logX = logY;
                acc = acc+1;
            end
            
            if mod(j,Mb)==0           % Stepsize adaptation.
                rat = acc/Mb;         % Estimate acceptance probability, and keep near 0.234.
                if rat>0.3
                    beta1=beta1*r;
                    beta2=beta2*r;
                elseif rat<0.1
                    beta1=beta1/r;
                    beta2=beta2/r;
                end
                acc=0;
            end
            
            if mod(j,THIN(k)) == 0
                samp(j/THIN(k),:) = x;     % Store samples.
            end
        end
        samplerTime(m,k) = toc;
        
        % Save quantities of interest.
        for n=1:numSamp
            rhoAsVec = paramToRhoCol(samp(n,:));
            rhoEst = reshape(rhoAsVec,[4 4]).';       % Watch map convention here.
            Fest(n) = real(PHI'*rhoEst*PHI);
        end
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

FileName = ['pcnLLdata_' datestr(Today,'yyyy') datestr(Today,'mm') ...
    datestr(Today,'dd') '_' fileNum];
save(FileName,'THIN','samplers','samplerTime','Fmean','Fstd','alpha')

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
% Converts length-36 parameter set (row vector) into density matrix (expressed as a column vector).
    function z = paramToRhoCol(par)
        Xr = reshape(par(1:16),[4 4]);
        Xi = reshape(par(17:32),[4 4]);
        X = Xr + i*Xi;                      % Matrix of column vectors (unnormalized).
        
        NORM = vecnorm(X,2,1);              % Compute the norm of each column.
        W = X./NORM;                        % Normalize each column.
       
        Y = par(33:36);                     % Projector weights.        
        gamma = Y/sum(Y);                   % Normalize.
        
        rho = W*diag(gamma)*W';             % Density matrix.
        z = reshape(rho.',[],1);            % Convert to column, in order of rho(1,1), rho(1,2), rho(1,3),...,etc.
    end

 end