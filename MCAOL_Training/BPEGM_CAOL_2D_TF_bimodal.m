function [ d1_res, d2_res, z1_res, z2_res, A1_dkxl_res, A2_dkxl_res, obj_val, iterations ] = ...
        BPEGM_CAOL_2D_TF_bimodal(x1, x2, size_kernel, gamma1, gamma2, lambda, arcdegree, ...
        M_type, max_it, tol, verbose, init_d1, init_d2, normZ)

%| BPEGM_CAOL_2D_TF_bimodal:
%| Multi-channel Convolutional Analysis Operator Learning (CAOL).
%|
%| [Input]
%| x1, x2: training images in sqrt(N) x sqrt(N) x L
%| size_kernel: [psf_s, psf_s, K]
%| gamma1, gamma2: regularization parameter for each channel sparsifying feature term
%| lambda: scaling param. for majorization matrix
%| arcdegree: param. for gradient-based restarting, within [90,100] 
%|         (default: 'H' in Prop. 5.1 of DOI: 10.1109/TIP.2019.2937734)
%| max_it: max number of iterations
%| tol: tolerance value for the relative difference stopping criterion
%| verbose: option to show intermidiate results
%| init: initial values for filters, sparse codes
%|
%| [Output]
%| d1_res, d2_res: jointly learned filters
%| z1_res, z2_res: final updates of sparse codes for each energy channel
%| A1_dkxl_res, A2_dkxl_res: final filtered images via learned sparsifying filters
%| obj_val: joint objective function value
%| iterations: records for iterations 
%|
%| ************************************************************************
%| A. Perelli, S. Alfonso Garcia, A. Bousse, J.-P. Tasu and N. Efthimiadis and D. Visvikis,
%| Multi-channel convolutional analysis operator learning for dual-energy CT reconstruction,
%| Physics in Medicine & Biology, 67(6), 065001, 2022
%| [Ref.] DOI: 10.1088/1361-6560/ac4c32
%| Copyright 2021-06-29, Alessandro Perelli, Suxer Alfonso Garcia, LaTIM, UMR 1101

%% Def: Parameters, Variables, and Operators
if size_kernel(1)*size_kernel(2) > size_kernel(3)
    error('The number of filters must be equal or larger than size of the filters.');
end
K = size_kernel(3);     %number of filters
L = size(x1,3);          %number of training images

%variables for filters
psf_radius = floor( [size_kernel(1)/2, size_kernel(2)/2] );

%dimensions of training images and sparse codes
size_x = [size(x1,1), size(x1,2), L];
size_z = [size(x1,1), size(x1,2), K, L];

%Coordinates on filters (only for odd-sized filters)
[kern_xgrid, kern_ygrid] = meshgrid(-psf_radius(1) : psf_radius(2), ...
                            -psf_radius(2) : psf_radius(2));

%Pad training images (default: circular boundary condition)
[~, x1pad] = PadFunc(x1, psf_radius);
[~, x2pad] = PadFunc(x2, psf_radius);

%Proximal operators for l0 "norm"
%ProxSparseL0 = @(u, theta) u .* ( abs(u) >= theta );   % to be changed later to generalised l0

%Convolutional operators
A1_k = @(u) A_for_dk( x1pad, u, size_x );
A1h_k = @(u) Ah_for_dk( x1pad, u, size_x, size_kernel );
A1_kl = @(u) A_for_dk_xl( x1pad, u, size_z );

A2_k = @(u) A_for_dk( x2pad, u, size_x );
A2h_k = @(u) Ah_for_dk( x2pad, u, size_x, size_kernel );
A2_kl = @(u) A_for_dk_xl( x2pad, u, size_z );


%Majorization matrix design for filter updating problems
disp('Pre-computing majorization matrices...');
if strcmp(M_type, 'I')
    %Scaled identity majorization matrix in
    %Lem. 5.2 of DOI: 10.1109/TIP.2019.2937734
    
    Md1 = majorMat_Ak_diag( x1pad, size_x, size_kernel );
    Md2 = majorMat_Ak_diag( x2pad, size_x, size_kernel );

elseif strcmp(M_type, 'H')
    %Exact Hessian matrix in
    %Prop. 5.1 of DOI: 10.1109/TIP.2019.2937734
    
    Md1 = majorMat_Ak_full( x1pad, size_x, size_kernel, L, psf_radius, kern_xgrid, kern_ygrid );
    Md2 = majorMat_Ak_full( x2pad, size_x, size_kernel, L, psf_radius, kern_xgrid, kern_ygrid );

else
    error('Choose an appropriate majorizer type.');
end
disp('Majorizer pre-computed...!');

%scaled majorization matrix
Ad1 = lambda * Md1;
Ad2 = lambda * Md2;

%adaptive restarting: Cos(theta), theta: angle between two vectors (rad)
omega = cos(pi*arcdegree/180);  

%Objective
%objective = @(z, A_dkxl) objectiveFunction( z, A_dkxl, alpha );

jointObjective = @(z1,z2,A1_dkxl,A2_dkxl) jointObjectiveFunction ( z1, z2, A1_dkxl, A2_dkxl, gamma1, gamma2 ) ;


%% Initialization
%Initialization: filters
if ~isempty(init_d1)
    d1 = init_d1;
else
    %Random initialization
    d1 = randn(size_kernel);
end

if ~isempty(init_d2)
    d2 = init_d2;
else
    %Random initialization
    d2 = randn(size_kernel);
end

%set the first filter as a DC filter
d1(:,:,1) = 1;
d2(:,:,1) = 1;
%filter normalization
for k = 1:K
    d1(:,:,k) = d1(:,:,k) ./ (sqrt(size_kernel(1)*size_kernel(2))*norm(d1(:,:,k),'fro'));
    d2(:,:,k) = d2(:,:,k) ./ (sqrt(size_kernel(1)*size_kernel(2))*norm(d2(:,:,k),'fro'));
end    
d1_p = d1;  
d2_p = d2;    

%Initialization: sparse codes
A1_dkxl = A1_kl(d1);  
A2_dkxl = A2_kl(d2);        
% z1 = ProxSparseL0( A1_dkxl, sqrt(2/gamma1) );
% z2 = ProxSparseL0( A2_dkxl, sqrt(2/gamma2) );

switch normZ
    case 'l0'
        [z1, z2] = genHardthres2D(A1_dkxl,A2_dkxl,gamma1,gamma2) ;
    case 'l1'
        if gamma1~=gamma2
            z1 = randn(size_z) ;
            z2 = randn(size_z) ;
            [z1, z2] = genSoftthres2D(z1, z2, A1_dkxl, A2_dkxl, gamma1, gamma2) ;
        else
            [z1, z2] = softthres2D(A1_dkxl, A2_dkxl, 1/gamma1) ;
        end
end




%ETC
tau = 1;            %momentum coeff.            
weight = 1-eps;     %delta in (7) of 10.1109/TIP.2019.2937734

%Save all objective values and timings
iterations.obj_vals = [];
iterations.tim_vals = [];
iterations.it_vals = [];

%Initial vals
%obj_val = objective(z, A_dkxl);
obj_val = jointObjective(z1,z2,A1_dkxl,A2_dkxl) ;
    
%Save all initial vars
iterations.obj_vals(1) = obj_val;
iterations.tim_vals(1) = 0;
%iterations.it_vals = cat(4, iterations.it_vals, d);

%Debug progress
fprintf('Iter %d, Obj %3.3g, Diff %5.5g\n', 0, obj_val, 0)

%Display initializations 
if verbose == 1
    iterate_fig1 = figure();
    filter_fig1 = figure();
    
    iterate_fig2 = figure();
    filter_fig2 = figure();
    
    display_func(iterate_fig1, filter_fig1, d1, z1, z1, x1, psf_radius(1), 0);
    display_func(iterate_fig2, filter_fig2, d2, z2, z2, x2, psf_radius(1), 0);
end
   
  
%% %%%%%%%%%% Two-block CAOL via reBPEG-M %%%%%%%%%%
for i = 1 : max_it
            
    %% UPDATE: All filters, { d_k : k=1,...,K }
    
    tic; %timing
    %%%%%%%%%%%%%%%%%%%%% reG-BPEG-M %%%%%%%%%%%%%%%%%%%%%%  
    if i ~= 1
        %Extrapolation with momentum!
        w_d = min( (tau_old - 1)/tau, weight*0.5*(lambda-1)/(lambda+1) );
        d1_p = d1 + w_d .* (d1 - d1_old);
        d2_p = d2 + w_d .* (d2 - d2_old);
    end
   
    %Proximal mapping
    d1_old = d1;
    d2_old = d2;
    
    Adnu1 = zeros(size_kernel);
    Adnu2 = zeros(size_kernel);
    
    for k = 1 : K
        if strcmp(M_type, 'I')
            Adnu1(:,:,k) = Ad1 .* d1_p(:,:,k) - A1h_k( A1_k(d1_p(:,:,k)) ...
                - reshape(z1(:,:,k,:), [size_z(1),size_z(2),size_z(4)]) ); 
            Adnu2(:,:,k) = Ad2 .* d2_p(:,:,k) - A2h_k( A2_k(d2_p(:,:,k)) ...
                - reshape(z2(:,:,k,:), [size_z(1),size_z(2),size_z(4)]) ); 
            
            
        elseif strcmp(M_type, 'H')
            AhA1_dp = Md1 * reshape(d1_p(:,:,k),[],1);
            AhA2_dp = Md2 * reshape(d2_p(:,:,k),[],1);
            
            Addp_m_AhAdp1 = (lambda-1) * AhA1_dp;
            Addp_m_AhAdp2 = (lambda-1) * AhA2_dp;
            
            Adnu1(:,:,k) = reshape(Addp_m_AhAdp1, [size_kernel(1), size_kernel(2)]) + ...
                A1h_k( reshape(z1(:,:,k,:), [size_z(1),size_z(2),size_z(4)]) );  
            Adnu2(:,:,k) = reshape(Addp_m_AhAdp2, [size_kernel(1), size_kernel(2)]) + ...
                A2h_k( reshape(z2(:,:,k,:), [size_z(1),size_z(2),size_z(4)]) );  
        else
            error('Choose an appropriate majorizer type.');
        end
    end
    d1 = ProxFilterTightFrame( Adnu1, size_kernel );
    d2 = ProxFilterTightFrame( Adnu2, size_kernel );
    
    %Gradient-based adaptive restarting
    if strcmp(M_type, 'I')
        Ad_diff1 = repmat(Ad1,[1,1,K]) .* (d1-d1_old);
        Ad_diff2 = repmat(Ad2,[1,1,K]) .* (d2-d2_old);
    elseif strcmp(M_type, 'H')
        Ad_diff1 = reshape( Ad1 * reshape(d1-d1_old, [size_kernel(1)*size_kernel(2), size_kernel(3)]), size_kernel );
        Ad_diff2 = reshape( Ad2 * reshape(d2-d2_old, [size_kernel(1)*size_kernel(2), size_kernel(3)]), size_kernel );
    else
        error('Choose an appropriate majorizer type.');
    end
    
    if dot( d1_p(:)-d1(:), Ad_diff1(:) ) / ( norm(d1_p(:)-d1(:)) * norm(Ad_diff1(:)) ) > omega
        d1_p = d1;
        Adnu1 = zeros(size_kernel);
        for k = 1 : K
            if strcmp(M_type, 'I')
                Adnu1(:,:,k) = Ad1 .* d1_p(:,:,k) - A1h_k( A1_k(d1_p(:,:,k)) ...
                    - reshape(z1(:,:,k,:), [size_z(1),size_z(2),size_z(4)]) );
            elseif strcmp(M_type, 'H')
                AhA1_dp = Md2 * reshape(d1_p(:,:,k),[],1);
                Addp_m_AhAdp1 = (lambda-1) * AhA1_dp;
                Adnu1(:,:,k) = reshape(Addp_m_AhAdp1, [size_kernel(1), size_kernel(2)]) + ...
                    A1h_k( reshape(z1(:,:,k,:), [size_z(1),size_z(2),size_z(4)]) );  
            else
                error('Choose an appropriate majorizer type.');
            end
        end
        d1 = ProxFilterTightFrame( Adnu1, size_kernel );
        disp('Restarted: filter update! (d1)');
    end
    
    if dot( d2_p(:)-d2(:), Ad_diff2(:) ) / ( norm(d2_p(:)-d2(:)) * norm(Ad_diff2(:)) ) > omega
        d2_p = d2;
        Adnu2 = zeros(size_kernel);
        for k = 1 : K
            if strcmp(M_type, 'I')
                Adnu2(:,:,k) = Ad2 .* d2_p(:,:,k) - A2h_k( A2_k(d2_p(:,:,k)) ...
                    - reshape(z2(:,:,k,:), [size_z(1),size_z(2),size_z(4)]) );
            elseif strcmp(M_type, 'H')
                AhA2_dp = Md2 * reshape(d2_p(:,:,k),[],1);
                Addp_m_AhAdp2 = (lambda-1) * AhA2_dp;
                Adnu2(:,:,k) = reshape(Addp_m_AhAdp2, [size_kernel(1), size_kernel(2)]) + ...
                    A2h_k( reshape(z2(:,:,k,:), [size_z(1),size_z(2),size_z(4)]) );
            else
                error('Choose an appropriate majorizer type.');
            end
        end
        d2 = ProxFilterTightFrame( Adnu2, size_kernel );
        disp('Restarted: filter update! (d2)');
    end
    
    
  
        
    %% UPDATE: All sparse codes, { z_{l,k} : l=1,...,L, k=1,...,K }
    
    %Proximal mapping with no majorization
    A1_dkxl = A1_kl(d1);
    A2_dkxl = A2_kl(d2);
    z1_old = z1;
    z2_old = z2;
    %z1 = ProxSparseL0( A1_dkxl, sqrt(2*alpha) );
    %[z1, z2] = genHardthres2D(A1_dkxl,A2_dkxl,gamma1,gamma2) ;
    
    switch normZ
        case 'l0'
            [z1, z2] = genHardthres2D(A1_dkxl,A2_dkxl,gamma1,gamma2) ;
        case 'l1'
            if gamma1~=gamma2
                [z1, z2] = genSoftthres2D(z1, z2, A1_dkxl, A2_dkxl, gamma1, gamma2) ;
            else
                [z1, z2] = softthres2D(A1_dkxl, A2_dkxl, 1/gamma1) ;
            end
    end
    
    
    
    
    %% UPDATE: Momentum coeff.  
    tau_old = tau;
    tau = ( 1 + sqrt(1 + 4*tau^2) ) / 2;
    
    %timing
    t_update = toc;

    
    %% EVALUATION    
    %Debug process
    %[obj_val, ~, ~] = objective(z, A_dkxl);
    obj_val = jointObjective(z1,z2,A1_dkxl,A2_dkxl) ;
    
    d1_relErr = norm( d1(:)-d1_old(:) ) / norm( d1(:) );
    z1_relErr = norm( z1(:)-z1_old(:) ) / norm( z1(:) );
    dnorm1 = norm(d1(:),2)^2;     
    
    d2_relErr = norm( d2(:)-d2_old(:) ) / norm( d2(:) );
    z2_relErr = norm( z2(:)-z2_old(:) ) / norm( z2(:) );
    dnorm2 = norm(d2(:),2)^2;     
    
    
    
    fprintf('Iter %d, Obj %3.3g, Filt. Norm 1 %2.2g, Filt. Norm 2 %2.2g,  D1 Diff %5.5g, D2 Diff %5.5g, Z1 Diff %5.5g,  Z2 Diff %5.5g\n', ...
              i, obj_val, dnorm1, dnorm2, d1_relErr, d2_relErr, z1_relErr, z2_relErr);
    
    %Record current iteration
    iterations.obj_vals(i + 1) = obj_val;
    iterations.tim_vals(i + 1) = iterations.tim_vals(i) + t_update;
%     if mod(i,500) == 0  %save filters every 500 iteration
%        iterations.it_vals = cat(4, iterations.it_vals, d); 
%     end
    
    %Display intermediate results 
    if verbose == true
        if mod(i,50) == 1
            display_func(iterate_fig1, filter_fig1, d1, A1_dkxl, z1, x1, psf_radius(1), i);
            display_func(iterate_fig2, filter_fig2, d2, A2_dkxl, z2, x2, psf_radius(1), i);
        end
    end
    
    %Termination
     if (d1_relErr < tol) && (z1_relErr < tol) && (d2_relErr < tol) && (z2_relErr < tol) && (i > 1)
         disp('relErr reached'); 
         break;
     end
    
end

%Final estimate    
d1_res = d1;
z1_res = z1;
A1_dkxl_res = A1_dkxl;

d2_res = d2;
z2_res = z2;
A2_dkxl_res = A2_dkxl;



return;



%%%%%%%%%%%%%%%%%% Added by A. Bousse %%%%%%%%%%%%%%%%%%%

function [z1, z2] = genSoftthres2D(z1, z2, y1, y2, c1, c2)
% min of 0.5*c1*(Z1 - Y1)^2 + 0.5*c2*(Z2 - Y2)^2 +  sqrt(Z1^2 + Z2^2)

c = max(c1, c2) ;

for i = 1 : 10  % only 10 iterations because it's slow.... why?
    
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


    z1(isnan(z1)) = 0 ;   % this stem is slow... how can it be speeded up?
    z2(isnan(z2)) = 0 ;



return;



function [Z1, Z2] = genHardthres2D(Y1,Y2,c1,c2)
% min of 0.5*c1*(Z1 - Y1)^2 + 0.5*c2*(Z2 - Y2)^2 + ( abs(Z1) + abs(Z2) ~= 0  )
    a = (0.5*c1.*Y1.^2 + 0.5*c2.*Y2.^2>= 1) ;
    Z1 = Y1.*a ;
    Z2 = Y2.*a ;
return


function [f_val, f_d, sparsity] = jointObjectiveFunction( z1, z2, A1_dkxl, A2_dkxl, gamma1, gamma2 )

    %Dataterm
    f_d1 = 0.5 * gamma1 * norm(A1_dkxl(:) - z1(:), 2)^2;
    f_d2 = 0.5 * gamma2 * norm(A2_dkxl(:) - z2(:), 2)^2;
    f_d = f_d1 + f_d2 ;
    %Regularizer
    f_z = nnz(abs(z1) + abs(z2) );

    %Function val
    f_val = f_d + f_z;
    
    %Sparsity
    sparsity = 100*f_z/numel(z1);
    
return;




%%%%%%%%%%%%%%%%%%%% Def: Padding Operators %%%%%%%%%%%%%%%%%%%%

function [M, bpad] = PadFunc(b, psf_radius)
    
M = padarray(ones(size(b)), [psf_radius(1), psf_radius(2), 0], 0, 'both');    %mask
%%%circular padding
bpad = padarray(b, [psf_radius(1), psf_radius(2), 0], 'circular', 'both');
%%%reflective padding
% bpad = padarray(b, [psf_radius(1), psf_radius(2), 0], 'symmetric', 'both');     
%%%%zero padding
% bpad = padarray(b, [psf_radius, psf_radius, 0], 0, 'both');              
    
return;




%%%%%%%%%%%%%%%%%%%% Def: System Operators %%%%%%%%%%%%%%%%%%%%

function Au = A_for_dk( xpad, u, size_x )

Au = zeros(size_x);
for l = 1 : size_x(3)
    %!!!NOTE: compensate rotating filter in conv2()
    Au(:,:,l) = conv2( xpad(:,:,l), rot90(u,2), 'valid' );
end

return;


function Ahu = Ah_for_dk( xpad, u, size_x, size_kernel )

Ahu = zeros(size_kernel(1), size_kernel(2)); 
for l = 1: size_x(3)
    Ahu = Ahu + conv2( xpad(:,:,l), rot90(u(:,:,l),2), 'valid');    
end

return;


function x_filt = A_for_dk_xl( xpad, d, size_z )

x_filt = zeros(size_z);
for l = 1 : size_z(4)
    for k = 1 : size_z(3)
        %!!!NOTE: compensate rotating filter in conv2()
        x_filt(:,:,k,l) = conv2( xpad(:,:,l), rot90(d(:,:,k),2), 'valid' );
    end
end

return;




%%%%%%%%%%%%%%%%%%%% Design: Majorization Matrices %%%%%%%%%%%%%%%%%%%%

function M =  majorMat_Ak_diag( xpad, size_x, size_kernel )
%Scaled identity majorization matrix in 
%Prop. 4.2 of DOI: 10.1109/TIP.2019.2937734


AtA_symb = zeros(size_kernel(1), size_kernel(2));
for l = 1 : size_x(3)
    P1x = xpad( 1 : 1+size_x(1)-1, 1 : 1+size_x(2)-1, l );
    for r2 = 1 : size_kernel(2)
        for r1 = 1 : size_kernel(1)
            Prx = xpad( r1 : r1+size_x(1)-1, r2 : r2+size_x(2)-1, l );
            AtA_symb(r1, r2) = AtA_symb(r1, r2) + Prx(:)' * P1x(:);
        end
    end
end

M = ( abs(AtA_symb(:))' * ones(size_kernel(1)*size_kernel(2),1) ) .* ...
    ones(size_kernel(1), size_kernel(1));

return;


function M = majorMat_Ak_full( xpad, size_x, size_kernel, L, psf_radius, kern_xgrid, kern_ygrid )  
%Exact Hessian matrix in
%Prop. 4.1 of DOI: 10.1109/TIP.2019.2937734


%!!!NOTE: x-grid and y-grid are horizontal and vertical direction, resp.
%E.g. in matlab, X( y-grid indices, x-grid indices )
kern_xgrid_vec = kern_xgrid(:);
kern_ygrid_vec = kern_ygrid(:);

M = zeros( size_kernel(1)*size_kernel(2), size_kernel(1)*size_kernel(2) );

for k1 = 1 : size_kernel(1)*size_kernel(2)
    for k2 = 1 : k1
        
        k1x_coord = kern_xgrid_vec(k1);
        k1y_coord = kern_ygrid_vec(k1);
        
        k2x_coord = kern_xgrid_vec(k2);
        k2y_coord = kern_ygrid_vec(k2);
        
        for l = 1 : L               
            xpad_k1 = xpad( 1 + psf_radius(1) + k1y_coord : size_x(1) + psf_radius(1) + k1y_coord, ...
                1 + psf_radius(2) + k1x_coord : size_x(2) + psf_radius(2) + k1x_coord, l );

            xpad_k2 = xpad( 1 + psf_radius(1) + k2y_coord : size_x(1) + psf_radius(1) + k2y_coord, ...
                1 + psf_radius(2) + k2x_coord : size_x(2) + psf_radius(2) + k2x_coord, l );

            if k1 == k2
                M(k1, k2) = M(k1, k2) + (xpad_k1(:)'*xpad_k2(:))/2;
            else
                M(k1, k2) = M(k1, k2) + xpad_k1(:)'*xpad_k2(:);
            end            
        end
    end
end

M = M + M';

return;




%%%%%%%%%%%%%%%%%%%% Def: Proximal Operators %%%%%%%%%%%%%%%%%%%%

function d = ProxFilterTightFrame( Adnu, size_kernel )
%Solve orthogonal Procrustes problem

kernVec_size = size_kernel(1)*size_kernel(2);

AdNu = reshape(Adnu, [kernVec_size, size_kernel(3)]);
[U, ~, V] = svd(AdNu, 'econ');

D = sqrt(kernVec_size)^(-1) * U * V';

d = reshape( D, size_kernel );

return;

    

    
function [] = display_func(iterate_fig, filter_fig, d, xfilt_d, z, x, psf_radius, iter)

    figure(iterate_fig);
    subplot(4,6,1), imshow(x(:,:,2),[]); axis image; colormap gray; title(sprintf('Local iterate %d',iter));
    
    subplot(4,6,2), imshow(xfilt_d(:,:,1,2),[]); axis image; colormap gray; title('Filt. img');
    subplot(4,6,3), imshow(xfilt_d(:,:,6,2),[]); axis image; colormap gray;
    subplot(4,6,4), imshow(xfilt_d(:,:,11,2),[]); axis image; colormap gray;
    subplot(4,6,5), imshow(xfilt_d(:,:,16,2),[]); axis image; colormap gray;
    subplot(4,6,6), imshow(xfilt_d(:,:,21,2),[]); axis image; colormap gray;
        
    subplot(4,6,8), imshow(z(:,:,1,2),[]); axis image; colormap gray; title('Spar. code'); 
    subplot(4,6,9), imshow(z(:,:,6,2),[]); axis image; colormap gray;
    subplot(4,6,10), imshow(z(:,:,11,2),[]); axis image; colormap gray;
    subplot(4,6,11), imshow(z(:,:,16,2),[]); axis image; colormap gray;
    subplot(4,6,12), imshow(z(:,:,21,2),[]); axis image; colormap gray;
    
    subplot(4,6,13), imshow(x(:,:,4),[]); axis image; colormap gray; title(sprintf('Local iterate %d',iter));
    
    subplot(4,6,14), imshow(xfilt_d(:,:,1,4),[]); axis image; colormap gray; title('Filt. img');
    subplot(4,6,15), imshow(xfilt_d(:,:,6,4),[]); axis image; colormap gray;
    subplot(4,6,16), imshow(xfilt_d(:,:,11,4),[]); axis image; colormap gray;
    subplot(4,6,17), imshow(xfilt_d(:,:,16,4),[]); axis image; colormap gray;
    subplot(4,6,18), imshow(xfilt_d(:,:,21,4),[]); axis image; colormap gray;

    subplot(4,6,20), imshow(z(:,:,1,4),[]); axis image; colormap gray; title('Spar. code'); 
    subplot(4,6,21), imshow(z(:,:,6,4),[]); axis image; colormap gray;
    subplot(4,6,22), imshow(z(:,:,11,4),[]); axis image; colormap gray;
    subplot(4,6,23), imshow(z(:,:,16,4),[]); axis image; colormap gray;
    subplot(4,6,24), imshow(z(:,:,21,4),[]); axis image; colormap gray;
    drawnow;

    figure(filter_fig);
    sqr_k = ceil(sqrt(size(d,3)));
    pd = 1;
    d_disp = zeros( sqr_k * [psf_radius*2+1 + pd, psf_radius*2+1 + pd] + [pd, pd]);
    for j = 0:size(d,3) - 1
        d_curr = d(:,:,j+1);
        d_disp( floor(j/sqr_k) * (size(d_curr,1) + pd) + pd + (1:size(d_curr,1)) , mod(j,sqr_k) * (size(d_curr,2) + pd) + pd + (1:size(d_curr,2)) ) = d_curr;
    end
    imagesc(d_disp); colormap gray; axis image; colorbar; 
    title(sprintf('Filter iterate %d',iter));
    drawnow;
        
return;
