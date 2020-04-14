 function pcnLL
% This function is designed to run many samplers performing Bayesian
% quantum state tomography on the experimental data from [Optica 5, 1455
% (2018)].

% It utilizes: (i) pCN Metropolis-Hastings and (ii) a pseudo-likelihood
% centered on the linear-inversion estimate.

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
alpha = 1;          % Parameters for gamma distribution prior.

Mb = 500;           % Update stepsize after this many points.
r = 1.1;            % Acceptance rate update factor.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CENTER OF PL DISTRIBUTION
proj = [1   1   0   1   1   1   0   1   0   0   0   0   1   1   0   1].';   % Only X&Z basis measurements are meaningful.

pauli0 = zeros(16,1);                 pauli0(1) = 1/4;                  % Results from linear inversion.
pauli0(2) = 0.003841866568897;        pauli0(4) = -0.014092279049696;
pauli0(5) = 0.000592418228315;        pauli0(6) = 0.222006753552445;
pauli0(8) = 0.008495387397081;        pauli0(13) = -0.004233814711591;
pauli0(14) = 0.009702721452970;       pauli0(16) = -0.231648675479682;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PROJECTION OPERATOR
% 16x16 Transformation from density to Pauli basis (see PRA 93.062320):
%                                     INDEX MAP (m,n)                           PAULI COEFFICIENTS (k,l) 
%         11  12  13  14  21  22  23  24  31  32  33  34  41  42  43  44    rho = z(k,l) [sigma(k)] x [sigma(l)]
U = 1/4*[  1   0   0   0   0   1   0   0   0   0   1   0   0   0   0   1  ; ...     % 00
           0   1   0   0   1   0   0   0   0   0   0   1   0   0   1   0  ; ...     % 01
           0   i   0   0  -i   0   0   0   0   0   0   i   0   0  -i   0  ; ...     % 02
           1   0   0   0   0  -1   0   0   0   0   1   0   0   0   0  -1  ; ...     % 03
           0   0   1   0   0   0   0   1   1   0   0   0   0   1   0   0  ; ...     % 10
           0   0   0   1   0   0   1   0   0   1   0   0   1   0   0   0  ; ...     % 11
           0   0   0   i   0   0  -i   0   0   i   0   0  -i   0   0   0  ; ...     % 12
           0   0   1   0   0   0   0  -1   1   0   0   0   0  -1   0   0  ; ...     % 13
           0   0   i   0   0   0   0   i  -i   0   0   0   0  -i   0   0  ; ...     % 20
           0   0   0   i   0   0   i   0   0  -i   0   0  -i   0   0   0  ; ...     % 21
           0   0   0  -1   0   0   1   0   0   1   0   0  -1   0   0   0  ; ...     % 22
           0   0   i   0   0   0   0  -i  -i   0   0   0   0   i   0   0  ; ...     % 23
           1   0   0   0   0   1   0   0   0   0  -1   0   0   0   0  -1  ; ...     % 30
           0   1   0   0   1   0   0   0   0   0   0  -1   0   0  -1   0  ; ...     % 31
           0   i   0   0  -i   0   0   0   0   0   0  -i   0   0   i   0  ; ...     % 32
           1   0   0   0   0  -1   0   0   0   0  -1   0   0   0   0   1 ];         % 33

T = proj.*U;   % Setting to zero all elements not measured by experiment.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DATA
% Counts for all basis combinations (coincidences only):
counts = [     7    304    280      8 ...       % ZZ
             151    128    154    159 ...       % ZX
             143    147    135    159 ...       % XZ
             289     18     12    297 ];        % XX

N = sum(counts);            % Total number of observations.
sigma = 1/sqrt(N);          % Standard deviation of pseudo-likelihood.

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
        logX = -1/(2*sigma^2)*4*norm(T*rhoX-pauli0)^2 ...
            + sum(alpha*log(x(33:36)) - x(33:36));      % PL & correction factor.
        
        % pCN Loop
        tic
        for j=1:numSamp*THIN(k)
            % Proposed updated parameters:
            newGauss = sqrt(1-beta1^2)*x(1:32) + beta1*randn(1,32);
            newGamma = x(33:36).*exp(beta2*randn(1,4));
            y = [newGauss newGamma];
            
            rhoY = paramToRhoCol(y);
            logY = -1/(2*sigma^2)*4*norm(T*rhoY-pauli0)^2 ...
                + sum(alpha*log(y(33:36)) - y(33:36));
            
            if log(rand) < logY - logX
                x = y;      % Accept new point.
                logX = logY;
                acc = acc+1;
            end
            
            if mod(j,Mb)==0         % Stepsize adaptation
                rat = acc/Mb;       % Estimate acceptance probability, and keep near 0.234
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

FileName = ['pcnPLdata_' datestr(Today,'yyyy') datestr(Today,'mm') ...
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