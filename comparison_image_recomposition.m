rng(0);
%% Load image
A = imread('lincoln.png');
A = im2gray(A);
A = im2double(A);
[m,n] = size(A);
normA_full = norm(A,'fro');

%% Define three mask types: uniform, blob, scratch

masks = struct();

% Uniform random: 40% observed
p_keep = 0.4;
M_uniform = rand(m,n) < p_keep;
masks(1).name = 'uniform';
masks(1).M    = M_uniform;

% Blob mask: random smooth blobs missing
noise = rand(m,n);
blob_field = imgaussfilt(noise, 10);
blob_thresh = 0.5;
blobMask = blob_field > blob_thresh;   % 1 = blob
M_blob = ~blobMask; 
masks(2).name = 'blob';
masks(2).M    = M_blob;

% Scratch mask: several thick scratches missing
M_scratch = true(m,n);
num_scratches = 6;
for k = 1:num_scratches
    y_start = randi(m);
    x_start = randi(n);
    y_end   = randi(m);
    x_end   = randi(n);

    num_pts = max(abs(y_end - y_start), abs(x_end - x_start)) + 1;
    ys = round(linspace(y_start, y_end, num_pts));
    xs = round(linspace(x_start, x_end, num_pts));

    for t = 1:num_pts
        yy = ys(t); xx = xs(t);
        y_min = max(1, yy-1); y_max = min(m, yy+1);
        x_min = max(1, xx-1); x_max = min(n, xx+1);
        M_scratch(y_min:y_max, x_min:x_max) = false;
    end
end
masks(3).name = 'scratch';
masks(3).M    = M_scratch;

% Hyperparameter grids
ranks_grid    = [20 40 60 80 100];       % for naive + ALS
lambda_grid   = [0.05 0.1 0.2 0.5 1.0];  % for nuclear norm
tau           = 1.0;
maxIter_nn    = 2000;
tol_nn        = 1e-4;
maxIter_lr    = 200;    % ALS iterations

results_all = struct();

numMasks = numel(masks);
colors = lines(numMasks);

nn_iters_cell = cell(numMasks,1);
nn_err_cell   = cell(numMasks,1);
als_iters_cell = cell(numMasks,1);
als_err_cell   = cell(numMasks,1);


for m_ix = 1:numel(masks)
    mask_name = masks(m_ix).name;
    M         = masks(m_ix).M;

    fprintf('\n==============================\n');
    fprintf('Mask type: %s\n', mask_name);
    fprintf('==============================\n');

    A_obs = M .* A;
    mask_missing   = ~M;
    normA_missing  = norm(mask_missing .* A, 'fro');

    % Naive (mean-fill + SVD) parameter sweep
    fprintf('\n[Naive] parameter sweep over ranks\n');
    best_naive_err = inf;
    best_naive_idx = 1;
    naive_relErr   = zeros(numel(ranks_grid),1);
    naive_rank_eff = zeros(numel(ranks_grid),1);

    tic;
    [X_naive_list, info_naive] = image_recomposition_naive_function( ...
        A_obs, M, ranks_grid, A);
    t_naive_sweep = toc;

    for k = 1:numel(ranks_grid)
        Xr = X_naive_list{k};
        relErr = norm(A - Xr, 'fro') / normA_full;
        naive_relErr(k) = relErr;

        s = svd(Xr,'econ');
        naive_rank_eff(k) = nnz(s > 1e-4);

        if relErr < best_naive_err
            best_naive_err = relErr;
            best_naive_idx = k;
        end
    end

    best_rank_naive   = ranks_grid(best_naive_idx);
    X_naive_best      = X_naive_list{best_naive_idx};
    naive_err_missing = norm(mask_missing .* (A - X_naive_best), 'fro') / normA_missing;

    fprintf('Best Naive rank = %d | relErr_full = %.4f\n', ...
            best_rank_naive, best_naive_err);

    % Low-rank factorization (ALS) parameter sweep

        fprintf('\n[ALS] parameter sweep over ranks\n');
    best_lr_err  = inf;
    best_lr_idx  = 1;
    lr_relErr    = zeros(numel(ranks_grid),1);
    lr_rank_eff  = zeros(numel(ranks_grid),1);

    for k = 1:numel(ranks_grid)
        r = ranks_grid(k);

        U_init = [];
        V_init = [];

        tic;
        [U_lr, V_lr, info_lr] = image_recomposition_low_rank_factorization_function( ...
            A_obs, M, U_init, V_init, r, maxIter_lr, false);
        t_lr_k = toc;

        X_lr_k = U_lr * V_lr';
        relErr = norm(A - X_lr_k, 'fro') / normA_full;
        lr_relErr(k) = relErr;

        s = svd(X_lr_k,'econ');
        lr_rank_eff(k) = nnz(s > 1e-4);

        fprintf('  r = %3d | relErr_full = %.4f | time = %.2fs\n', r, relErr, t_lr_k);

        if relErr < best_lr_err
            best_lr_err = relErr;
            best_lr_idx = k;
            X_lr_best   = X_lr_k;
            best_U_lr   = U_lr;
            best_V_lr   = V_lr;
        end
    end

    best_rank_lr  = ranks_grid(best_lr_idx);
    lr_err_missing = norm(mask_missing .* (A - X_lr_best), 'fro') / normA_missing;

    fprintf('Best ALS rank = %d | relErr_full = %.4f\n', best_rank_lr, best_lr_err);

    % Nuclear norm (SVT) lambda sweep

    fprintf('\n[Nuclear norm] parameter sweep over lambda\n');
    best_nn_err   = inf;
    best_nn_idx   = 1;
    nn_relErr     = zeros(numel(lambda_grid),1);
    nn_rank_eff   = zeros(numel(lambda_grid),1);
    nn_runtimes   = zeros(numel(lambda_grid),1);

    for k = 1:numel(lambda_grid)
        lambda = lambda_grid(k);

        tic;
        [X_nn_k, info_nn_k] = image_recomposition_nuclear_norm_function( ...
            A_obs, M, lambda, [], ...
            'maxiter', maxIter_nn, ...
            'tol', tol_nn, ...
            'stepsize', tau, ...
            'verbose', false, ...
            'A_true', A);
        t_nn_k = toc;

        relErr = norm(A - X_nn_k, 'fro') / normA_full;
        nn_relErr(k)   = relErr;
        nn_runtimes(k) = t_nn_k;

        s = svd(X_nn_k,'econ');
        nn_rank_eff(k) = nnz(s > 1e-4);

        fprintf('  lambda = %.3f | relErr_full = %.4f | rank = %d | time = %.2fs\n', ...
                lambda, relErr, nn_rank_eff(k), t_nn_k);

        if relErr < best_nn_err
            best_nn_err = relErr;
            best_nn_idx = k;
            X_nn_best   = X_nn_k;
            best_info_nn = info_nn_k;
        end
    end

    best_lambda_nn   = lambda_grid(best_nn_idx);
    nn_err_missing   = norm(mask_missing .* (A - X_nn_best), 'fro') / normA_missing;

    fprintf('Best lambda = %.3f | relErr_full = %.4f\n', best_lambda_nn, best_nn_err);

    %% Store best results for this mask
 
    
    results_all(m_ix).mask_name = mask_name;
    results_all(m_ix).M         = M;
    results_all(m_ix).A_obs     = A_obs;
    
    results_all(m_ix).naive.rank        = best_rank_naive;
    results_all(m_ix).naive.X           = X_naive_best;
    results_all(m_ix).naive.err_full    = best_naive_err;
    results_all(m_ix).naive.err_missing = naive_err_missing;
    
    results_all(m_ix).lr.rank        = best_rank_lr;
    results_all(m_ix).lr.X           = X_lr_best;
    results_all(m_ix).lr.err_full    = best_lr_err;
    results_all(m_ix).lr.err_missing = lr_err_missing;
    
    results_all(m_ix).nn.lambda      = best_lambda_nn;
    results_all(m_ix).nn.X           = X_nn_best;
    results_all(m_ix).nn.info        = best_info_nn;
    results_all(m_ix).nn.err_full    = best_nn_err;
    results_all(m_ix).nn.err_missing = nn_err_missing;

    %% Record convergence curves

    % Nuclear norm convergence
    [X_nn_conv, info_nn_conv] = image_recomposition_nuclear_norm_function( ...
        A_obs, M, best_lambda_nn, [], ...
        'maxiter', 200, ...
        'tol', 1e-6, ...
        'stepsize', tau, ...
        'verbose', false, ...
        'A_true', A);
    
    nn_iters_cell{m_ix} = (1:numel(info_nn_conv.relErr_full))';
    nn_err_cell{m_ix}   = info_nn_conv.relErr_full(:);
    
    % ALS convergence
    [U_lr_conv, V_lr_conv, info_lr_conv] = image_recomposition_low_rank_factorization_function( ...
        A_obs, M, [], [], best_rank_lr, maxIter_lr, false);
    
    als_iters_cell{m_ix} = (1:numel(info_lr_conv.relError))';
    als_err_cell{m_ix}   = info_lr_conv.relError(:);


    
    %% ---------------------------------------------------------
    %  Visualization: side-by-side for this mask
    %% --------------------------------------------------------
    figure('Name',sprintf('Reconstructions - %s mask', mask_name),'NumberTitle','off');

    subplot(1,5,1);
    imshow(A, []); title('Original');

    subplot(1,5,2);
    imshow(A_obs + 1.*(~M), []); title(sprintf('Observed (%s)', mask_name));

    subplot(1,5,3);
    imshow(X_naive_best, []);
    title(sprintf('Naive (r=%d)\nerr=%.3f', best_rank_naive, best_naive_err));

    subplot(1,5,4);
    imshow(X_lr_best, []);
    title(sprintf('ALS (r=%d)\nerr=%.3f', best_rank_lr, best_lr_err));

    subplot(1,5,5);
    imshow(X_nn_best, []);
    title(sprintf('Nuclear (λ=%.3f)\nerr=%.3f', best_lambda_nn, best_nn_err));
end

figure('Name','ALS vs Nuclear norm convergence','NumberTitle','off');
hold on;

for ix = 1:numMasks
    res = results_all(ix);

    % Nuclear Norm 
    plot(nn_iters_cell{ix}, nn_err_cell{ix}, ...
         'LineWidth', 2, ...
         'Color', colors(ix,:), ...
         'LineStyle', '-', ...
         'DisplayName', sprintf('%s - nuclear', res.mask_name));

    % ALS 
    plot(als_iters_cell{ix}, als_err_cell{ix}, ...
         'LineWidth', 2, ...
         'Color', colors(ix,:), ...
         'LineStyle', '--', ...
         'DisplayName', sprintf('%s - ALS', res.mask_name));
end

xlabel('Iterations');
ylabel('Relative Error (full image)');
title('ALS vs Nuclear norm convergence for different masks');
grid on;
legend('Location','northeast');
xlim([1, 200]);
hold off;