%% Load image
A = imread('lincoln.png');
A = im2gray(A);
A = double(A)/255;

%% Lambda sweep for nuclear norm
lambdas = [10 5 2];
svt = run_nuclear_norm_function(A, lambdas);
all_ranks = svt.ranks(:);
nuc_err = svt.errors(:);
nuc_time = svt.runtimes(:);

%% Run truncated SVD at the same ranks induced by nuclear norm
tsvd = run_truncated_svd_function(A, all_ranks);
svd_err  = tsvd.errors(:);
svd_time = tsvd.runtimes(:);

%% Build comparison table
Comp = table( ...
    lambdas(:), ...
    all_ranks(:), ...
    svd_err(:), ...
    nuc_err(:), ...
    svd_time(:), ...
    nuc_time(:), ...
    'VariableNames', ...
    {'Lambda','Rank','SVD_Error','Nuc_Error','SVD_Time','Nuc_Time'} ...
);

disp('=== Comparison Table (Lambda sweep: 2, 5, 10) ===');
disp(Comp);

%% Show the reconstructed images
numShow = numel(lambdas);

figure;
for j = 1:numShow
    rj      = all_ranks(j);
    lambdaj = lambdas(j);

    Asvd_j = tsvd.solutions{j};

    Asvt_j = svt.solutions{j};

    subplot(2, numShow, j);
    imshow(Asvd_j, []);
    title(sprintf('SVD (r=%d)\nerr=%.3f', rj, svd_err(j)));

    subplot(2, numShow, numShow + j);
    imshow(Asvt_j, []);
    title(sprintf('\\lambda=%g\nr=%d\nerr=%.3f', lambdaj, rj, nuc_err(j)));
end

sgtitle('Lowest-Rank Reconstructions: SVD vs Nuclear Norm');