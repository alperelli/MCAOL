function [x1, x2, z1, z2] = recoCT_CAOL_dualEnergy(y1, y2, d1, d2, paramCT1, paramCT2, paramReco, paramFilters)

%| recoCT_CAOL_dualEnergy:
%| Multi-channel Convolutional Analysis Operator Learning (CAOL) Dual-Energy CT (DECT) reconstruction .
%|
%| [Input]
%| y1, y2: dual energy projection measurements in N_d x N_{theta}
%| d1, d2: jointly trained filters through MCAOL
%| paramCT1, paramCT2: geometrical CT parameters for generating DECT measurements;
%| paramReco: struct which includes the free parameters (channel likelihood weights parameters.rho1, parameters.rho2) in the MCAOL reconstruction function and LBFGS solver (max number of iterations)
%| paramFilter: parameters struct of the learned jointly filters ( param.gamma1 and param.gamma2): regularization parameter for sparse code terms)
%|
%| [Output]
%| x1, x2: jreconstructed DECT images
%| z1, z2: final estimates of sparse codes for each energy  
%|
%| ************************************************************************
%| A. Perelli, S. Alfonso Garcia, A. Bousse, J.-P. Tasu and N. Efthimiadis and D. Visvikis,
%| Multi-channel convolutional analysis operator learning for dual-energy CT reconstruction,
%| Physics in Medicine & Biology, 67(6), 065001, 2022
%| [Ref.] DOI: 10.1088/1361-6560/ac4c32
%| Copyright 2021-06-29, Alessandro Perelli, Suxer Alfonso Garcia, LaTIM, UMR 1101

%% parameters and initialization
N = paramReco.imSize(1) ;
x1 = ones(N,N)/2 ;
x2 = ones(N,N)/2 ;

size_kernel = size(d1) ;
psf_radius = floor( [size_kernel(1)/2, size_kernel(2)/2] );
size_z = [N,N,size(d1,3)] ;

gamma1 = paramFilters.gamma1 ;
gamma2 = paramFilters.gamma2 ;

rho1 = paramReco.rho1 ;
rho2 = paramReco.rho2 ;


% the following  is because x1 and x2 use different scales for projecting

disp('initialization: image')

% Options for L-BFGS-B
opts.maxIts = paramReco.nIterInit ;
opts.printEvery = 1 ;
opts.verbose = 0;
opts.m = 9 ;
opts.factr = 1e7 ;


x1vect = x1(:) ;
lb = zeros(size(x1vect)) ;
ub = 1000000*ones(size(x1vect)) ;
opts.x0 = x1vect ;
[x1vect,~,~] = lbfgsb(@(x1vect)log_lik(x1vect,y1,paramCT1) , lb, ub, opts) ;
x1 = reshape(x1vect,N,N) ;

x2vect = x2(:) ;
lb = zeros(size(x2vect)) ;
ub = 1000000*ones(size(x2vect)) ;
opts.x0 = x2vect ;
[x2vect,~,~] = lbfgsb(@(x2vect)log_lik(x2vect,y2,paramCT2) , lb, ub, opts) ;
x2 = reshape(x2vect,N,N) ;


figure(1)

subplot(1,2,1)
imagesc(x1) ; axis image ;
colormap gray
title('x1: initialisation')

subplot(1,2,2)
imagesc(x2) ; axis image ;
colormap gray
title('x2: initialisation')

pause(5), close;

dkx1 = fconv(x1,d1, size_z, psf_radius) ;  
dkx2 = fconv(x2,d2, size_z, psf_radius) ;

[z1, z2] = genHardthres2D(dkx1,dkx2,gamma1,gamma2) ;


switch paramFilters.norm
    case 'l0'
        [z1, z2] = genHardthres2D(dkx1,dkx2,gamma1,gamma2) ;
    case 'l1'
        if gamma1~=gamma2
            z1 = randn(size_z) ;
            z2 = randn(size_z) ;
            [z1, z2] = genSoftthres2D(z1, z2, dkx1, dkx2, gamma1, gamma2) ;
        else
            [z1, z2] = softthres2D(dkx1, dkx2, 1/gamma1) ;
        end
end


% Options for L-BFGS-B
opts.maxIts = paramReco.nInnerIter ;

norm_z1_old = sum(z1(:)~=0) ;
norm_z2_old = sum(z2(:)~=0) ;

for i = 1 : paramReco.nIterMax
    
    %% image update
    % image 1
    x1vect = x1(:) ;
    opts.x0 = x1vect ;
    [x1vect,~,~] = lbfgsb(@(x1vect)fullcost(x1vect,y1,d1,z1, paramCT1, size_z, psf_radius,rho1,gamma1) , lb, ub, opts) ;
    x1 = reshape(x1vect,N,N) ;
    
    % image 2
    x2vect = x2(:) ;
    opts.x0 = x2vect ;
    [x2vect,~,~] = lbfgsb(@(x2vect)fullcost(x2vect,y2,d2,z2, paramCT2, size_z, psf_radius,rho2,gamma2) , lb, ub, opts) ;
    x2 = reshape(x2vect,N,N) ;
    
    %% sparse  update
    dkx1 = fconv(x1,d1, size_z, psf_radius) ;
    dkx2 = fconv(x2,d2, size_z, psf_radius) ;
    
    switch paramFilters.norm
        case 'l0'
            [z1, z2] = genHardthres2D(dkx1,dkx2,gamma1,gamma2) ;
        case 'l1'
            if gamma1~=gamma2
                [z1, z2] = genSoftthres2D(z1, z2, dkx1, dkx2, gamma1, gamma2) ;
            else
                [z1, z2] = softthres2D(dkx1, dkx2, 1/gamma1) ;
            end
    end
      
    norm_z1_new = sum(z1(:)~=0) ;
    disp(['z1 l0-norm: ',num2str(norm_z1_new ), ' (previously ', num2str(norm_z1_old)]) ;
    norm_z1_old = norm_z1_new ;
    
    norm_z2_new = sum(z2(:)~=0) ;
    disp(['z2 l0-norm: ',num2str(norm_z2_new ), ' (previously ', num2str(norm_z2_old)]) ;
    norm_z2_old = norm_z2_new ;
    
    % analysis ...............
    
    if mod(i, 10) == 0
        figure(2); 
        subplot(1,2,1), imagesc(x1,[0,1]), colormap gray ;
        axis image, title(['recon x1, iteration ',num2str(i)]) ;
        subplot(1,2,2), imagesc(x2,[0,1]), colormap gray ; 
        axis image, title(['recon x2, iteration ',num2str(i)]) ;
    end
    
    figure(3)
    subplot(1,5,1), imagesc(z1(:,:,1)) , axis image, colormap gray, title('z1');
    subplot(1,5,2), imagesc(z1(:,:,6)) , axis image, colormap gray
    subplot(1,5,3), imagesc(z1(:,:,11)), axis image, colormap gray
    subplot(1,5,4), imagesc(z1(:,:,16)), axis image, colormap gray
    subplot(1,5,5), imagesc(z1(:,:,21)), axis image, colormap gray
    
    figure(4)
    subplot(1,5,1), imagesc(z2(:,:,1)) , axis image, colormap gray, title('z2')
    subplot(1,5,2), imagesc(z2(:,:,6)) , axis image, colormap gray
    subplot(1,5,3), imagesc(z2(:,:,11)), axis image, colormap gray
    subplot(1,5,4), imagesc(z2(:,:,16)), axis image, colormap gray
    subplot(1,5,5), imagesc(z2(:,:,21)), axis image, colormap gray
    
    pause(0.5);   
    
end    


return






function [f,g] = fullcost(xvect,y,d,z, paramReco, size_z, psf_radius,rho,gamma)

[f1,g1] = log_lik(xvect,y,paramReco) ;
[f2,g2] = sparse_cost(xvect,d,z, paramReco, size_z, psf_radius) ;

f = f1 + (gamma/rho)*f2 ;
g = g1 + (gamma/rho)*g2 ;

return

function [f,g] = sparse_cost(xvect,d,z, paramReco, size_z, psf_radius)

x = reshape(xvect,paramReco.imSize) ;
diff = fconv(x,d, size_z, psf_radius) - z ;
grad = bconv(diff,d, size_z, psf_radius) ;
f = 0.5 * sum( diff(:).^2 ) ;
g = grad(:) ;

return



function [f,g] = log_lik(xvect,y,paramReco)

x = reshape(xvect,paramReco.imSize) ;
exp_proj = exp( - forwardProj(x,paramReco) );
ybar = paramReco.I.*exp_proj +  paramReco.bckg   ;

f = y.*log(ybar) - ybar ;
f = - sum(f(:)) ;

ratio = y./ybar ;
g =  backProj(paramReco.I.*exp_proj.*  (1 - ratio),paramReco)  ;
g = -g(:) ;


return

function [z1, z2] = genSoftthres2D(z1, z2, y1, y2, c1, c2)
% min of 0.5*c1*(Z1 - Y1)^2 + 0.5*c2*(Z2 - Y2)^2 +  sqrt(Z1^2 + Z2^2)

c = max(c1, c2) ;

for i = 1 : 10  
    
    z1_tmp = z1 - c1*(z1-y1)./c ;
    z2_tmp = z2 - c2*(z2-y2)./c ;
    
    [z1, z2] = softthres2D(z1_tmp,z2_tmp,1./c) ;
    
end


return



function [z1, z2] = softthres2D(y1, y2, alpha)
    % min of 0.5*(Z1 - Y1)^2 + 0.5*(Z2 - Y2)^2 + gamma * sqrt(Z1^2 + Z2^2)
    normY1Y2 = sqrt(y1.^2 + y2.^2 ) ;

    M = max(normY1Y2 - alpha,0) ;

    z1 = M.*y1./normY1Y2 ;
    z2 = M.*y2./normY1Y2 ;


    z1(isnan(z1)) = 0 ; 
    z2(isnan(z2)) = 0 ;



return;




function [Z1, Z2] = genHardthres2D(Y1,Y2,c1,c2)
% min of 0.5*c1*(Z1 - Y1)^2 + 0.5*c2*(Z2 - Y2)^2 + ( abs(Z1) + abs(Z2) ~= 0  )
    a = (0.5*c1.*Y1.^2 + 0.5*c2.*Y2.^2>= 1) ;
    Z1 = Y1.*a ;
    Z2 = Y2.*a ;
return



function RES = fconv(x,d, size_z, psf_radius)

[~, xpad] = PadFunc(x, psf_radius);

RES = A_for_dk_x( xpad, d, size_z );

return


function RES = bconv(x,d, size_z, psf_radius)

[~, xpad] = PadFunc(x, psf_radius);

RES = sum(  Ah_for_dk_diff_k(xpad, d, size_z),3  ) ;

return




function [M, bpad] = PadFunc(b, psf_radius)
    
M = padarray(ones(size(b)), [psf_radius(1), psf_radius(2), 0], 0, 'both');    %mask
%%%circular padding
bpad = padarray(b, [psf_radius(1), psf_radius(2), 0], 'circular', 'both');
%%%reflective padding
% bpad = padarray(b, [psf_radius(1), psf_radius(2), 0], 'symmetric', 'both');     
%%%%zero padding
% bpad = padarray(b, [psf_radius, psf_radius, 0], 0, 'both');              
    
return;

function x_filt = A_for_dk_x( xpad, d, size_z )

x_filt = zeros(size_z);

for k = 1 : size_z(3)
    %!!!NOTE: compensate rotating filter in conv2()
    x_filt(:,:,k) = conv2( xpad, rot90(d(:,:,k),2), 'valid' );
end


return;





function diff_filt = Ah_for_dk_diff_k( diffpad, d, size_z )
%B = B(:,:,1) ;
diff_filt = zeros(size_z(1),size_z(2),size_z(3));



for k = 1 : size_z(3)
    %!!!NOTE: compensate rotating filter in conv2()
    
    
    back_conv = conv2(diffpad(:,:,k),d(:,:,k), 'valid' ) ;
    
    %diff_filt(:,:,k,l) = reshape(back_conv(B==1),[size_z(1),size_z(2)]  ) ;
    diff_filt(:,:,k) = back_conv ; %reshape(back_conv(B==1),[size_z(1),size_z(2)]  ) ;
end


return;
