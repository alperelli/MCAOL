function sino = forwardProj(im, paramProj)
% General forward projector

N = size(im,1) ;
sigma_mm = paramProj.FWHM/2.3555 ;

sigma_vox = sigma_mm/paramProj.voxSize ;

phi = 180*paramProj.phi/pi ;
nPhi = length(phi) ;
nPSF = round(N/10) ;

sino = zeros(N,nPhi) ;

if paramProj.GPU == 1
    im = gpuArray(single(im)) ;
    sino = gpuArray(single(sino)) ;
end


if (mod(nPSF,2)==0)
    h = fspecial('gaussian',[1 nPSF-1],sigma_vox) ;
else
    h = fspecial('gaussian',[1 nPSF],sigma_vox) ;
end


H = {h,h} ;
im = convnsep(H,im,'same') ;

for i = 1 : nPhi
    
    im_rot = imrotate(im,phi(i),'bilinear','crop') ;
    sino(:,i) = (sum(im_rot,1))' ;
    
end

sino = paramProj.time * sino ;

if paramProj.GPU == 1
    sino = double(gather(sino)) ;
end





