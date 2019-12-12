function v = ToVector_K_dimention(im,k)
% takes MxNx3 picture and returns (MN)x3 vector
sz = size(im);
v = reshape(im, [prod(sz(1:2)) k]);