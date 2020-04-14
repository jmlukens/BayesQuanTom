function simulatedData
% This function generates data for a simulated two-qudit state, computing
% the LS density matrix estimate and probability matrix to be used for
% Bayesian QST.

% Joseph M. Lukens (lukensjm@ornl.gov)
% 2020.04.14
% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ %
close all;  clear all;
%% INITIAL PARAMETERS:
d = 5;              % Single-qudit dimensionality.
lambda = 0.95;      % Werner state settings (to add noise).
N = 100*d^2;        % Number of counts for each basis setting.
fileNum = '001';    

% Weyl generators:
fftMatrix = d*ifft(eye(d));
Z = diag(fftMatrix(2,:));
X = circshift(eye(d),1);

% Generalized Pauli matrices for single qudit:
pauli = zeros(d,d,d^2);
for j=1:d
    for k=1:d
        if j==1 && k==1
            pauli(:,:,(j-1)*d+k) = eye(d);
        elseif j==k
            r = j-1;
            pauli(:,:,(j-1)*d+k) = sqrt(2/(r*(r+1)))*diag([ones(1,r) -r zeros(1,d-1-r)]);
        elseif j<k
            test = zeros(d,d);
            test(j,k) = 1;
            test(k,j) = 1;
            pauli(:,:,(j-1)*d+k) = test;
        elseif k<j
            test = zeros(d,d);
            test(j,k) = i;
            test(k,j) = -i;
            pauli(:,:,(j-1)*d+k) = test;
        end
    end
end

pauli2 = zeros(d^2,d^2,d^4);         % Array of all operators for two qudits.
for j=1:d^2
    for k=1:d^2
        pauli2(:,:,(j-1)*d^2+k) = kron(pauli(:,:,j),pauli(:,:,k));
    end
end

% Ground truth state:
psi0 = zeros(d,1);
for j=1:d
    psi0((j-1)*d+j) = 1/sqrt(d);
end
rho0 = lambda*psi0*psi0' + (1-lambda)/d^2*eye(d^2);
rho0vec = reshape(rho0.',[d^4 1]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% MEASUREMENT SETTINGS:
M = [];                 % Measurement matrix (in computational basis).
R = [];                 % Measurement matrix (in Pauli basis).
r = zeros(1,d^2-1);

% Bases considered for each qudit (in terms of Weyl powers):
xA = [1 0 ones(1,d-1)];         % This should produce a complete set of MUBs.
zA = [0 1 (1:1:d-1)];

xB = [1 0 ones(1,d-1)];
zB = [0 1 (1:1:d-1)];

for m = 1:length(xA)
    [A ~] = eig(X^xA(m)*Z^zA(m));           % Basis states.
    for n = 1:length(xB)
        [B ~] = eig(X^xB(n)*Z^zB(n));
        for p=1:d
            for q=1:d
                psi = kron(A(:,p),B(:,q));      % Joint state.
                vec = kron(conj(psi),psi);      % Measurement vector (computational basis).
                M = [M; vec.'];

                for j=1:d^4-1                   % Expressed in Pauli operator space (skip identity part).
                    r(j) = psi'*pauli2(:,:,j+1)*psi;
                end
                R = [R; real(r)];
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SIMULATED DATA
counts = [];        freq=[];
prob = M*rho0vec;

for m = 1:length(xA)*length(xB)
    pMeas = real(prob((m-1)*d^2+1:m*d^2));        % Extracting relevant probabilities.
    pMeas(pMeas<0) = 0;
    countsNew = mnrnd(N,pMeas).';                 
    counts = [counts; countsNew];
    freq = [freq; countsNew/N];                  % Expressed as empirical frequencies.
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% LS ESTIMATE
c = zeros(d^4,1);           % Pauli coefficients.
c(1) = 1/d^2;               % Normalization coefficient (ensures trace=1).
c(2:d^4) = R\(freq-c(1));   % LS estimate of remaining coefficients.

% Final LS matrix:
rhoLS = zeros(d^2,d^2);
for j=1:d^4
    rhoLS = rhoLS + c(j)*pauli2(:,:,j);
end

% Computing quantities of interest:
fprintf(['Fidelity of LS estimate: ' num2str(real(psi0'*rhoLS*psi0)) '\n'])
fprintf(['LS eigenvalues: ' num2str(eig(rhoLS).') '\n'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SAVE RESULTS TO FILE
Today = date;
fileName = ['simData_' datestr(Today,'yyyy') datestr(Today,'mm') ...
    datestr(Today,'dd') '_' fileNum];
save(fileName,'N','psi0','lambda','M','counts','rhoLS')

