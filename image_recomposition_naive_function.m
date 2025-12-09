function [X_list, info] = image_recomposition_naive_function(A_obs, M, ranks, A_true)
    tic;

    [m, n] = size(A_obs);
    ranks  = ranks(:)'; 
    R      = numel(ranks);

    % Compute mean of observed pixels
    observed_vals = A_obs(M ~= 0);
    mean_val = mean(observed_vals(:));

    % Build the filled matrix
    A_tilde = A_obs;
    A_tilde(M == 0) = mean_val;             % fill missing with mean

    % SVD of the filled matrix
    [U,S,V] = svd(A_tilde, 'econ');
    
    X_list = cell(R,1);

    info.ranks      = ranks;
    info.mean_value = mean_val;
    info.relError   = [];
    info.absError   = [];

    have_true = (nargin >= 4 && ~isempty(A_true));
    if have_true
        nA = norm(A_true, 'fro');
        info.relError = zeros(R,1);
        info.absError = zeros(R,1);
    end

    % Build rank-r reconstructions for each requested rank
    for k = 1:R
        r = ranks(k);
        r = min(r, size(U,2));

        Ur = U(:,1:r);
        Sr = S(1:r,1:r);
        Vr = V(:,1:r);

        Xr = Ur * Sr * Vr'; 
        X_list{k} = Xr;

        if have_true
            err = norm(A_true - Xr, 'fro');
            info.absError(k) = err;
            info.relError(k) = err / nA;
        end
    end

    info.runtime = toc;
end