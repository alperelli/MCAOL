function showFilters(d,param)

figure()

pd = 1;
sqr_k = ceil(sqrt(size(d,3)));
d_disp = zeros( sqr_k * [param.size_kernel(1) + pd, param.size_kernel(2) + pd] + [pd, pd]);
for j = 0:size(d,3) - 1
    d_disp( floor(j/sqr_k) * (param.size_kernel(1) + pd) + pd + (1:param.size_kernel(1)),...
        mod(j,sqr_k) * (param.size_kernel(2) + pd) + pd + (1:param.size_kernel(2)) ) = d(:,:,j + 1); 
end
imagesc(d_disp); colormap gray; axis image;  colorbar; title('Final filter estimate');

end