function im = backProj(sino,paramProj)
% General back projector

N = size(sino,1) ;
sigma_mm = paramProj.FWHM/2.3555 ;
sigma_vox = sigma_mm/paramProj.voxSize ;

phi = 180*paramProj.phi/pi ;

nPhi = length(phi) ; % now nPhi is the size of the subset....
nPSF = round(N/5) ;

if (mod(nPSF,2)==0)
    h = fspecial('gaussian',[1 nPSF-1],sigma_vox) ;
else
    h = fspecial('gaussian',[1 nPSF],sigma_vox) ;
end

H = {h,h} ;
im = zeros(N,N) ;

if paramProj.GPU == 1
    im = gpuArray(single(im)) ;
    sino = gpuArray(single(sino)) ;
end

for i = 1 : nPhi
    
    r_sino = repmat( (sino(:,i))', [N, 1]   ) ;
    im = im + imrotate(r_sino,-phi(i),'bilinear','crop') ;
    
end

im = convnsep(H,im,'same') ;
im = paramProj.time * im ;


if paramProj.GPU == 1
    im = double(gather(im)) ;
end



    