
%% MCAOL for Training _- Sparse Dual Energy CT
%| A. Perelli, S. Alfonso Garcia, A. Bousse, J.-P. Tasu and N. Efthimiadis and D. Visvikis,
%| Multi-channel convolutional analysis operator learning for dual-energy CT reconstruction,
%| Physics in Medicine & Biology, 67(6), 065001, 2022
%| [Ref.] DOI: 10.1088/1361-6560/ac4c32
%| Copyright 2021-06-29, Alessandro Perelli, Suxer Alfonso Garcia, LaTIM, UMR 1101
%| ************************************************************************

clear; close all; clc;

xcat = load('../Data/XCAT_DE.mat') ;
 
mu_60 = xcat.mu_60 ;
mu_60 = mu_60(:,:,[1:12,14:25]) ; 

param.max_attn_60 = max(mu_60(:));
x1                 = mu_60/max(param.max_attn_60);

 mu_120 = xcat.mu_120 ;
mu_120 = mu_120(:,:,[1:12,14:25]) ;
 
param.max_attn_120 = max(mu_120(:));
x2                  = mu_120/max(param.max_attn_120);
param.study = 'CT_60_120' ;


%% Parameters
%Type of filter regularizers in CAOL (default: 'tf')
param.reg = 'tf';    %option: 'tf', 'div'

%Hyperparameters
param.size_kernel = [7, 7, 49]; 

%Majorization matrix options (default: 'H')
%|Read descriptions of "M_type" in the following function: "BPEGM_CAOL_2D_TF.m" 
param.major_type = 'H';   

%Options for BPEG-M algorithms
param.lambda = 1+eps;    %scaling param. for majorization matrix (default: 1+eps)
param.arcdegree = 90;    %param. for gradient-based restarting: angle between two vectors 
                                          %(default: 90 degree)
param.max_it =1000 ;      %max number of iterations
param.tol    = 1e-4;          %tol. val. for the relative difference stopping criterion

%Fixed random initialization
%s = RandStream('mt19937ar','Seed',1);
%RandStream.setGlobalStream(s);

%Initial filters
%| If [], then the CDL codes will automatically initialize them.
init_d1 = randn(param.size_kernel);    %no need to normalize
init_d2 = randn(param.size_kernel); 

%Display intermediate results?
verbose_disp = 1;    %option: 1, 0

%Save results?
saving = 1;   %option: 1, 0

param.norm = 'l0' ;

scale = 1;

for i = scale
    
    param.gamma1 = 1e3; 
    param.gamma2 = param.gamma1;

    % param.norm = 'l1' ;
    % % gamma1 and gamma2 should be equal
    % param.gamma1 = 80 ; 
    % param.gamma2 = param.gamma1 ;


    %% CAOL via BPEG-M
    fprintf('CAOL with %d x [%d x %d] kernels.\n\n', ...
        param.size_kernel(3), param.size_kernel(1), param.size_kernel(2) )

    [ d1, d2, z1, z2, x1_filt, x2_filt, obj, iterations ] = BPEGM_CAOL_2D_TF_bimodal(x1, x2, param.size_kernel, ...
            param.gamma1, param.gamma2, param.lambda, param.arcdegree, param.major_type, ...
            param.max_it, param.tol, verbose_disp, init_d1, init_d2, param.norm );

    %Save results
    if saving == true
        if 7~=exist ('saved_filters', 'dir')
            mkdir ('saved_filters')
        end
        
        gamma1 = num2str(param.gamma1);
        save(['saved_filters/filters_',param.study,'_',param.norm ,'.mat'], 'param', 'init_d1', 'd1', 'init_d2', 'd2','obj', 'iterations');
        fprintf('Data saved.\n');
    end
    
end
   
