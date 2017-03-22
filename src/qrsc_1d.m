clear
clc
close all

% Initialization
n = 2^11; % # grid points
r = 20; % domain [0,r] 
h = r / n;
x = linspace(0, r, n)'; % grid points
L = -2 * eye(n) + diag(ones(n-1, 1), 1) + diag(ones(n-1, 1), -1);
L(1, n) = 1; L(n, 1) = 1; % periodic Laplacian


nV = 5;
vmean = r * rand(nV, 1); % Gaussian centers
vmean = [vmean; 0; r];
%vmean = linspace(0,r,nV);
V = zeros(n, 1);
for i = 1:nV + 2
    V = V + normpdf(x, vmean(i), 1);
end
V = diag(V);
H = -0.5 * L + V; % operator

% Precomputation
N = 250; % # orbits
[Psi, e] = eig(H, 'vector');
Psi_N = Psi(:, 1:N); % orbits
% pairwise Hadamard product
Rho = repmat(Psi_N', N, 1) .* reshape(repmat(Psi_N, N, 1), n, N^2)';

% QR Selected-Column
tic;
M = fft(bsxfun(@times, Rho, exp(2 * pi * 1i * rand(N^2, 1))));
ind = randperm(N^2);
M = M(ind(1:20*N),:);
[Q, R, p] = qr(M, 0);
toc

eps = 1e-5;
Naux = nnz(abs(diag(R))>eps * abs(R(1, 1)));
P = R(1:Naux, 1:Naux)\R(1:Naux, :);
P(:, p) = P;
Rho_qr = Rho(:, p(1:Naux))*P;

% Llyod's Selected-Column
tic;
w = sum(Psi_N.^4, 2); % weights (quadratic/quartic)
idx = w > (1e-1 * max(w)); % filtering
opt.weight = w(idx);
%opt.careful = 1;
[~, C, ~] = fkmeans(x(idx), min(150+Naux, nnz(idx)), opt); % weighted k-means
C = unique(max(1, ceil(C/h)));
Rho_vo = Rho(:, C)*(Rho(:, C)\Rho); % least-square 
toc
s = max(w(p(1:Naux)))/min(w(p(1:Naux)));

% Print relative error for approximation
fprintf('Approximation rel-err for qr is %.8f.\n',norm(Rho_qr-Rho,'fro')/norm(Rho,'fro'));
fprintf('Approximation rel-err for vo is %.8f.\n',norm(Rho_vo-Rho,'fro')/norm(Rho,'fro'));

% Plot different interpolation points
plot(x, w)
hold on
plot(x(p(1:Naux)), w(p(1:Naux)),'ro')
plot(x(C), w(C),'g*')