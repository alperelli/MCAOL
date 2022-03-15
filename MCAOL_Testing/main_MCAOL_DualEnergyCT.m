
%% Main CAOL dual energy reconstruction with LBFGS
%| A. Perelli, S. Alfonso Garcia, A. Bousse, J.-P. Tasu, N. Efthimiadis, D. Visvikis,
%| Multi-channel convolutional analysis operator learning for dual-energy CT reconstruction,
%| Physics in Medicine & Biology, 67(6), 065001, 2022
%| [Ref.] DOI: 10.1088/1361-6560/ac4c32
%| Copyright 2021-06-29, Alessandro Perelli, Suxer Alfonso Garcia, LaTIM, UMR 1101
%| ************************************************************************
clear all, close all; clc;

%% Load the paths for the tools

% misc tools
addpath(genpath('../Utils'))

%% load XCAT phantom
xcat        = load('../Data/XCAT_DE.mat') ;

mu_120 = xcat.mu_120 ;
mu_120 = mu_120(:,:,13) ; % slice 13 is for reconstruction

mu_60 = xcat.mu_60 ;
mu_60 = mu_60(:,:,13) ;   % slice 13 is for reconstruction

% Normalize the images
param.max_attn_120 = max(mu_120(:)) ;
param.max_attn_60  = max(mu_60(:)) ;

x1    = mu_120 / max(param.max_attn_120) ;
x1_gt = x1     / max(abs(x1(:))) ;
x2    = mu_60  / max(param.max_attn_60) ;
x2_gt = x2     / max(abs(x2(:))) ;

N = size(x1_gt,1) ;


%% Parameters CT geometry and forward operator
paramCT1.phi     = 0 : 0.1 : 2*pi ; % 63 angles (used for sparse view reconstruction)
%paramCT1.phi     = 0 : 0.05 : 2*pi ; % 126 angles (used for low counts reconstruction)
paramCT1.voxSize =  1.0; %mm 
paramCT1.FWHM    =  0.5; 
paramCT1.I       = 1e5 ; % source intensity (used for sparse view reconstruction)
%paramCT1.I       = 5e3 ; % source intensity (used for low counts reconstruction)
paramCT1.GPU     = 1 ;
paramCT1.bckg    = 0 ;   % paramReco.I*10 ;
paramCT1.imSize  = [N,N] ;

paramCT2         = paramCT1 ;

paramCT1.scale = param.max_attn_60 ;
paramCT2.scale = param.max_attn_120 ;

paramCT1.time = param.max_attn_60 ;
paramCT2.time = param.max_attn_120 ;


%% Generate Dual energy sinogram data
y1bar = paramCT1.I.*exp( -forwardProj(x1_gt,paramCT1)) +  paramCT1.bckg ;

y2bar = paramCT2.I.*exp( -forwardProj(x2_gt,paramCT2)) +  paramCT2.bckg ;

y1 = poissrnd(y1bar) ;
y2 = poissrnd(y2bar) ;


%% Parameters Reconstruction algorithm geometry and forward operator
paramReco = struct() ;
param     = struct() ;
paramReco.imSize     = [N,N] ;  % also required here

% MCAOL solver parameters
paramReco.nIterMax   = 300 ;
paramReco.nIterInit  = 150 ;
paramReco.nInnerIter = 150 ;

KAPPA_MCAOL = 5*1e-5;

% MCAOL DE learned filters
param.study = 'CT_60_120' ;
param.norm = 'l0';
aa = load(['../MCAOL_Training/saved_filters/filters_',param.study,'_',param.norm ,'.mat']) ;
d1 = aa.d1 ;
d2 = aa.d2 ;
paramFilters_DE = aa.param ;

paramReco.rho1 = KAPPA_MCAOL * paramFilters_DE.gamma1 ; 
paramReco.rho2 = KAPPA_MCAOL * paramFilters_DE.gamma2 ;


%%  MCAOL Reconstruction
[x1j, x2j, z1j, z2j] = recoCT_CAOL_dualEnergy(y1, y2, d1, d2, paramCT1, paramCT2, paramReco, paramFilters_DE);


%% Save Results
 if 7~=exist ('DECT_MCAOL_reconstruction', 'dir')
            mkdir ('DECT_MCAOL_reconstruction')
 end
 
saveDir = 'DECT_MCAOL_reconstruction';
fileName = [saveDir,'/reco_MCAOL_DECT.mat'] ;
save(fileName, 'x1j', 'x2j', 'paramReco') ;

fprintf('Please, find the reconstructed images in "DECT_MCAOL_reconstruction" folder!!! \n')







