 function sliceLLcholesky
% This function runs a bunch of independent samplers for several values of
% THIN, using (i) slice sampling, (ii) the full likelihood, and (iii) the
% Cholesky density matrix parameterization.

% Joseph M. Lukens (lukensjm@ornl.gov)
% 2020.04.14
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ %
clear all;  close all;
%% LOOP PARAMETERS
THIN = 2.^(0:1:5);      % THIN values considered for each sampler.
samplers = 1;           % Number of independent samplers per THIN value.
fileNum = '001';        % Number used in saved file.
t=tic;

WIDTH = 100;            % Parameters for slice sampler.
BURN = 0;
%% INPUTS
numSamp = 2^10;         % Number of samples to obtain from MH.
numParam = 15;          % Number of parameters to find.

PHI = 1/sqrt(2)*[0 1 1 0]';     % Ideal state.

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
         
for k=1:length(THIN)
    for m=1:samplers
        rng('shuffle')
        
        param0 = rand(1,numParam);     % Initial seed (all values between 0 & 1 here).
        tmp=@(x)logPD(x);

        Fest = zeros(numSamp,1);
        
        % Slice sampling:
        tic;
        samp = slicesample(param0,numSamp,'logpdf',tmp,'width',WIDTH,'burnin',BURN,'thin',THIN(k));
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

FileName = ['sliceLLcholeskyData_' datestr(Today,'yyyy') datestr(Today,'mm') ...
    datestr(Today,'dd') '_' fileNum];
save(FileName,'THIN','samplers','samplerTime','Fmean','Fstd')

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
title(['Cholesky'])


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
        u1 = par(1);
        u2 = par(2);
        u3 = par(3);
        q21 = par(4);
        q31 = par(5);
        q32 = par(6);
        q41 = par(7);
        q42 = par(8);
        q43 = par(9);
        phi21 = par(10);
        phi31 = par(11);
        phi32 = par(12);
        phi41 = par(13);
        phi42 = par(14);
        phi43 = par(15);
        
        % Cholesky matrix:
        L11 = (u1)^(1/2);
        L21 = ((1-u1)*u2*q21)^(1/2)*exp(i*phi21);
        L22 = ((1-u1)*u2*(1-q21))^(1/2);
        L31 = ((1-u1)*(1-u2)*u3*q31)^(1/2)*exp(i*phi31);
        L32 = ((1-u1)*(1-u2)*u3*(1-q31)*q32)^(1/2)*exp(i*phi32);
        L33 = ((1-u1)*(1-u2)*u3*(1-q31)*(1-q32))^(1/2);
        L41 = ((1-u1)*(1-u2)*(1-u3)*q41)^(1/2)*exp(i*phi41);
        L42 = ((1-u1)*(1-u2)*(1-u3)*(1-q41)*q42)^(1/2)*exp(i*phi42);
        L43 = ((1-u1)*(1-u2)*(1-u3)*(1-q41)*(1-q42)*q43)^(1/2)*exp(i*phi43);
        L44 = ((1-u1)*(1-u2)*(1-u3)*(1-q41)*(1-q42)*(1-q43))^(1/2);
        
        L = [L11 0 0 0; L21 L22 0 0; L31 L32 L33 0; L41 L42 L43 L44];
        z = L*L';
        z = reshape(z.',[],1);      % Convert to column vector
     end

 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Computing derivative:
    function z = paramTo_g(par)
        u1 = par(1);
        u2 = par(2);
        u3 = par(3);
        q21 = par(4);
        q31 = par(5);
        q32 = par(6);
        q41 = par(7);
        q42 = par(8);
        q43 = par(9);
        phi21 = par(10);
        phi31 = par(11);
        phi32 = par(12);
        phi41 = par(13);
        phi42 = par(14);
        phi43 = par(15);
        
        % Each column is a derivative against a different parameter (starting @ zeroth):
        dL11 = [(u1)^(1/2) 1/2*u1^(-1/2) 0 0 0 0 0 0 0 0 0 0 0 0 0 0];
        
        dL21 = [((1-u1)*u2*q21)^(1/2)*exp(i*phi21) ...
            -1/2*(1-u1)^(-1/2)*(u2*q21)^(1/2)*exp(i*phi21) ...
            1/2*u2^(-1/2)*((1-u1)*q21)^(1/2)*exp(i*phi21) 0 ...
            1/2*q21^(-1/2)*((1-u1)*u2)^(1/2)*exp(i*phi21) 0 0 0 0 0 ...
            i*((1-u1)*u2*q21)^(1/2)*exp(i*phi21) 0 0 0 0 0];
        
        dL22 = [((1-u1)*u2*(1-q21))^(1/2) ...
            -1/2*(1-u1)^(-1/2)*(u2*(1-q21))^(1/2) ...
            1/2*u2^(-1/2)*((1-u1)*(1-q21))^(1/2) 0 ...
            -1/2*(1-q21)^(-1/2)*((1-u1)*u2)^(1/2) 0 0 0 0 0 0 0 0 0 0 0 0];
        
        dL31 = [((1-u1)*(1-u2)*u3*q31)^(1/2)*exp(i*phi31) ...
            -1/2*(1-u1)^(-1/2)*((1-u2)*u3*q31)^(1/2)*exp(i*phi31) ...
            -1/2*(1-u2)^(-1/2)*((1-u1)*u3*q31)^(1/2)*exp(i*phi31) ...
            1/2*u3^(-1/2)*((1-u1)*(1-u2)*q31)^(1/2)*exp(i*phi31) 0 ...
            1/2*q31^(-1/2)*((1-u1)*(1-u2)*u3)^(1/2)*exp(i*phi31) 0 0 0 0 0 ...
            i*((1-u1)*(1-u2)*u3*q31)^(1/2)*exp(i*phi31) 0 0 0 0];
        
        dL32 = [((1-u1)*(1-u2)*u3*(1-q31)*q32)^(1/2)*exp(i*phi32) ...
            -1/2*(1-u1)^(-1/2)*((1-u2)*u3*(1-q31)*q32)^(1/2)*exp(i*phi32) ...
            -1/2*(1-u2)^(-1/2)*((1-u1)*u3*(1-q31)*q32)^(1/2)*exp(i*phi32) ...
            1/2*u3^(-1/2)*((1-u1)*(1-u2)*(1-q31)*q32)^(1/2)*exp(i*phi32) 0 ...
            -1/2*(1-q31)^(-1/2)*((1-u1)*(1-u2)*u3*q32)^(1/2)*exp(i*phi32) ...
            1/2*q32^(-1/2)*((1-u1)*(1-u2)*u3*(1-q31))^(1/2)*exp(i*phi32) 0 0 0 0 0 ...
            i*((1-u1)*(1-u2)*u3*(1-q31)*q32)^(1/2)*exp(i*phi32) 0 0 0];
        
        dL33 = [((1-u1)*(1-u2)*u3*(1-q31)*(1-q32))^(1/2) ...
            -1/2*(1-u1)^(-1/2)*((1-u2)*u3*(1-q31)*(1-q32))^(1/2) ...
            -1/2*(1-u2)^(-1/2)*((1-u1)*u3*(1-q31)*(1-q32))^(1/2) ...
            1/2*u3^(-1/2)*((1-u1)*(1-u2)*(1-q31)*(1-q32))^(1/2) 0 ...
            -1/2*(1-q31)^(-1/2)*((1-u1)*(1-u2)*u3*(1-q32))^(1/2) ...
            -1/2*(1-q32)^(-1/2)*((1-u1)*(1-u2)*u3*(1-q31))^(1/2) 0 0 0 0 0 0 0 0 0];
        
        dL41 = [((1-u1)*(1-u2)*(1-u3)*q41)^(1/2)*exp(i*phi41) ...
            -1/2*(1-u1)^(-1/2)*((1-u2)*(1-u3)*q41)^(1/2)*exp(i*phi41) ...
            -1/2*(1-u2)^(-1/2)*((1-u1)*(1-u3)*q41)^(1/2)*exp(i*phi41) ...
            -1/2*(1-u3)^(-1/2)*((1-u1)*(1-u2)*q41)^(1/2)*exp(i*phi41) 0 0 0 ...
            1/2*q41^(-1/2)*((1-u1)*(1-u2)*(1-u3))^(1/2)*exp(i*phi41) 0 0 0 0 0 ...
            i*((1-u1)*(1-u2)*(1-u3)*q41)^(1/2)*exp(i*phi41) 0 0];
        
        dL42 = [((1-u1)*(1-u2)*(1-u3)*(1-q41)*q42)^(1/2)*exp(i*phi42) ...
            -1/2*(1-u1)^(-1/2)*((1-u2)*(1-u3)*(1-q41)*q42)^(1/2)*exp(i*phi42) ...
            -1/2*(1-u2)^(-1/2)*((1-u1)*(1-u3)*(1-q41)*q42)^(1/2)*exp(i*phi42) ...
            -1/2*(1-u3)^(-1/2)*((1-u1)*(1-u2)*(1-q41)*q42)^(1/2)*exp(i*phi42) 0 0 0 ...
            -1/2*(1-q41)^(-1/2)*((1-u1)*(1-u2)*(1-u3)*q42)^(1/2)*exp(i*phi42) ...
            1/2*q42^(-1/2)*((1-u1)*(1-u2)*(1-u3)*(1-q41))^(1/2)*exp(i*phi42) 0 0 0 0 0 ...
            i*((1-u1)*(1-u2)*(1-u3)*(1-q41)*q42)^(1/2)*exp(i*phi42) 0];
        
        dL43 = [((1-u1)*(1-u2)*(1-u3)*(1-q41)*(1-q42)*q43)^(1/2)*exp(i*phi43) ...
            -1/2*(1-u1)^(-1/2)*((1-u2)*(1-u3)*(1-q41)*(1-q42)*q43)^(1/2)*exp(i*phi43) ...
            -1/2*(1-u2)^(-1/2)*((1-u1)*(1-u3)*(1-q41)*(1-q42)*q43)^(1/2)*exp(i*phi43) ...
            -1/2*(1-u3)^(-1/2)*((1-u1)*(1-u2)*(1-q41)*(1-q42)*q43)^(1/2)*exp(i*phi43) 0 0 0 ...
            -1/2*(1-q41)^(-1/2)*((1-u1)*(1-u2)*(1-u3)*(1-q42)*q43)^(1/2)*exp(i*phi43) ...
            -1/2*(1-q42)^(-1/2)*((1-u1)*(1-u2)*(1-u3)*(1-q41)*q43)^(1/2)*exp(i*phi43) ...
            1/2*q43^(-1/2)*((1-u1)*(1-u2)*(1-u3)*(1-q41)*(1-q42))^(1/2)*exp(i*phi43) 0 0 0 0 0 ...
            i*((1-u1)*(1-u2)*(1-u3)*(1-q41)*(1-q42)*q43)^(1/2)*exp(i*phi43)];
            
        dL44 = [((1-u1)*(1-u2)*(1-u3)*(1-q41)*(1-q42)*(1-q43))^(1/2) ...
            -1/2*(1-u1)^(-1/2)*((1-u2)*(1-u3)*(1-q41)*(1-q42)*(1-q43))^(1/2) ...
            -1/2*(1-u2)^(-1/2)*((1-u1)*(1-u3)*(1-q41)*(1-q42)*(1-q43))^(1/2) ...
            -1/2*(1-u3)^(-1/2)*((1-u1)*(1-u2)*(1-q41)*(1-q42)*(1-q43))^(1/2) 0 0 0 ...
            -1/2*(1-q41)^(-1/2)*((1-u1)*(1-u2)*(1-u3)*(1-q42)*(1-q43))^(1/2) ...
            -1/2*(1-q42)^(-1/2)*((1-u1)*(1-u2)*(1-u3)*(1-q41)*(1-q43))^(1/2) ...
            -1/2*(1-q43)^(-1/2)*((1-u1)*(1-u2)*(1-u3)*(1-q41)*(1-q42))^(1/2) 0 0 0 0 0 0];
        
        % Zeroth order L matrix:
        L = [dL11(1) 0 0 0; dL21(1) dL22(1) 0 0; ...
            dL31(1) dL32(1) dL33(1) 0; dL41(1) dL42(1) dL43(1) dL44(1)]; 
    
        % Derivative of density matrix for all elements.
        dRho = zeros(4,4,15);     
        g = zeros(15,15);           % g measure matrix
        for j = 1:15
            dL = [dL11(j+1) 0 0 0; dL21(j+1) dL22(j+1) 0 0; ...
                dL31(j+1) dL32(j+1) dL33(j+1) 0; dL41(j+1) dL42(j+1) dL43(j+1) dL44(j+1)]; 
        
            dRho(:,:,j) = dL*L' + L*dL';
            for p=1:j
                g(j,p) = trace(dRho(:,:,j)*dRho(:,:,p));
                g(p,j) = g(j,p);        % Exploiting symmetry.
            end
        end
        
        z = g;      % Return matrix.
    end
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTION TO SAMPLE
    function z = logPD(param)
        
        % Enforcing parameter limits:
        if any(param<1e-300) | any(param(1:9)>1) | any(param(10:15)>2*pi) 
             z = -1e300;
             return
        end
                
        rhoVec = paramToRhoCol(param);          % Compute density matrix.
        logLL = counts * log(U*rhoVec);         % Likelihood.
        
        % Integration measure
        G = paramTo_g(param);     % Compute matrix.
        logMEAS = 1/2*log(det(G));

        % Complete posterior distribution:
        z = real(logLL + logMEAS);      % Take real part for stability.
    end
  
toc(t)
end