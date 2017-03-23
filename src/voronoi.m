% [C, mu] = voronoi(Psi, x, Naux, eps)
% [C, mu] = voronic(Psi, x)
%
% Density fitting using Voronoi tessellation to compute interpolation
% points and least-square to compute the coefficient.
%
% Input:
%   Psi: Columns are orbits. n*N matirx
%   x: Lattice discretization for the domain of orbits.
%   Naux: Number of interpolation points. Default: 4*N
%   eps: Thresholding on weights. Default:1e-2
% Output:
%   C: Coefficient for each column of Rho in auxiliary basis. Naux*n matrix
%   mu: Chosen interpolation points.

function [C, mu] = voronoi(Psi, x, Naux, eps)
    if nargin < 4,  eps = 1e-2;  end
    if nargin < 3,  Naux = 4 * size(Psi,2);  end
    
    % setup
    h = x(2)-x(1);
    w = sum(Psi.^4, 2); % weights (quadratic/quartic)
    idx = w > (eps * max(w)); % filtering
    Naux = min(Naux,nnz(idx));
    opt.weight = w(idx);
    
    % weighted k-mean
    [~, mu, ~] = fkmeans(x(idx), Naux, opt); % weighted k-means
    mu = unique(max(1, ceil(mu/h)));
    
    % least square
    b = (Psi(mu,:)*Psi').^2;
    R = chol(b(:,mu)+1e-10*eye(length(mu))); % Tikhonov
    C = R\(R'\b);
end