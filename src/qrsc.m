% [P, mu] = qrsc(Rho, eps)
%
% Density fitting using pivoted QR column selection
%
% Input:
%   Rho: Pairwise Hadamard product of orbits. N^2*n matirx.
%   eps: Threshold for choosing auxiliary basis. Default: 1e-5
% Output:
%   P: Auxiliray basis functions. N_aux*n matrix
%   mu: Chosen interpolation points.

function [P, mu] = qrsc(Rho, eps)
    if nargin < 3,  eps = 1e-5; end
    
    % random Fourier projection
    N2 = size(Rho, 1);
    M = fft(bsxfun(@times, Rho, exp(2 * pi * 1i * rand(N2, 1))));
    ind = randperm(N2);
    M = M(ind(1 : 20*sqrt(N2)), :);
    
    % pivoted qr
    [Q, R, p] = qr(M, 0);
    Naux = nnz(abs(diag(R))>eps * abs(R(1, 1)));
    
    % column selection
    P = real(R(1:Naux, 1:Naux)\R(1:Naux, :));
    P(:, p) = P;
    mu = p(1:Naux);
end