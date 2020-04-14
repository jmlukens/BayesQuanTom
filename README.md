# BayesQuanTom

This set of MATLAB source code performs Bayesian quantum state tomography utilizing an efficient parameterization and pCN Metropolis-Hastings sampling algorithm. The theory behind it is described in detail in [J. M. Lukens, K. J. H. Law, A. Jasra, and P. Lougovski, “A practical and efficient approach for Bayesian quantum state estimation” (link forthcoming)]. The codes are designed to reproduce the results in this reference. They can be modified as needed to match a given experiment, by defining the likelihood or pseudo-likelihood based on actual measurements and data.

To run as is, the codes require the Statistics and Machine Learning Toolbox, specifically for the  functions slicesample (https://www.mathworks.com/help/stats/slicesample.html) and randg (https://www.mathworks.com/help/stats/randg.html). Should this toolbox not be available, all the codes based on pCN sampling will still run: the only modification required is to change how the initial point is drawn. We have found it effective to replace the initial draw from randg with randn and square the results (to ensure positive values).

The codes are divided into two categories, those performing tomography on experimental data from https://doi.org/10.1364/OPTICA.5.001455, and those which use data generated by the code simulatedData.m (included). Following we provide an overview of the adjustable settings which were used for the results presented in the paper:

Fig. 2(a): sliceLL.m with THIN=2.^(0:1:7); samplers=100; alpha=1;

Fig. 2(b): pcnLL.m with THIN=2.^(0:1:9); samplers=100; alpha=1;

Fig. 4(a): pcnLL.m and pcnPL with THIN=2.^(0:1:9); samplers=100; alpha=1;

Fig. 4(b): pcnLL.m and pcnPL. mwith THIN=2.^(0:1:9); samplers=100; alpha=1/4;

Fig. 5: timeTestLL.m and timeTestPL.m with:  
dataFileName = 'simData_L=0.95_d=2.mat'; d=2;  
dataFileName = 'simData_L=0.95_d=3.mat'; d=3;  
dataFileName = 'simData_L=0.95_d=5.mat'; d=5;  
dataFileName = 'simData_L=0.95_d=7.mat'; d=7;  

Fig. 6: runTwoQuditPL.m with samplers=1 and the following conditions:  
dataFileName = 'simData_L=0.95_d=2.mat'; THIN=2^7;  
dataFileName = 'simData_L=0.95_d=3.mat'; THIN=2^10;  
dataFileName = 'simData_L=0.95_d=5.mat'; THIN=2^13;  
dataFileName = 'simData_L=0.95_d=7.mat'; THIN=2^14;  
dataFileName = 'simData_L=0.85_d=2.mat'; THIN=2^7;  
dataFileName = 'simData_L=0.85_d=3.mat'; THIN=2^9;  
dataFileName = 'simData_L=0.85_d=5.mat'; THIN=2^12;  
dataFileName = 'simData_L=0.85_d=7.mat'; THIN=2^12;  
dataFileName = 'simData_L=0.75_d=2.mat'; THIN=2^7;  
dataFileName = 'simData_L=0.75_d=3.mat'; THIN=2^8;  
dataFileName = 'simData_L=0.75_d=5.mat'; THIN=2^12;  
dataFileName = 'simData_L=0.75_d=7.mat'; THIN=2^12;  

Fig. A1(a): sliceLLcholesky.m with THIN=2.^(0:1:7); samplers=1;  
sliceLL.m with THIN=2.^(0:1:7); samplers=1; alpha=1;

Fig. A1(b): indLL.m with THIN=2.^(0:1:21); samplers=1; alpha=1;  
pcnLL.m with THIN=2.^(0:1:9); samplers=1; alpha=1;

Please contact Joseph Lukens at lukensjm@ornl.gov with any comments or questions. Thanks for stopping by!
