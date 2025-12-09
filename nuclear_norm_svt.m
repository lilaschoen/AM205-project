

function [X, info] = nuclear_norm_svt(A, lambda, varargin)
% Solve min_X 0.5||A - X||_F^2 + lambda * ||X||_* using proximal gradient / singular value thresholding.

    maxIter  = 200;
    tol      = 1e-4;
    tau      = 1.0; 
    verbose  = false;

    for k = 1:2:length(varargin)
        name = varargin{k};
        val  = varargin{k+1};
        switch lower(name)
            case 'maxiter'
                maxIter = val;
            case 'tol'
                tol = val;
            case 'stepsize'
                tau = val;
            case 'verbose'
                verbose = val;
            otherwise
                error('Unknown option %s', name);
        end
    end

    % Initialization
    X = zeros(size(A));
    info.obj       = zeros(maxIter,1);
    info.errHist(k) = norm(A - X, 'fro') / max(1e-12, norm(A, 'fro'));
    info.relChange = zeros(maxIter,1);
    info.rank      = zeros(maxIter,1);

    for k = 1:maxIter
        X_old = X;

        % Gradient step: Y = X - tau*(X - A)
        Y = X - tau * (X - A);

        % SVD of Y
        [U,S,V] = svd(Y, 'econ');
        s = diag(S);

        % Soft-threshold singular values
        s_shrunk = max(s - tau*lambda, 0);

        % Keep only nonzero singular values to save work
        r = nnz(s_shrunk > 0);
        if r == 0
            X = zeros(size(A));
        else
            U_r = U(:,1:r);
            V_r = V(:,1:r);
            S_r = diag(s_shrunk(1:r));
            X = U_r * S_r * V_r';
        end

        info.rank(k) = r;
        info.obj(k)  = 0.5*norm(A - X,'fro')^2 + lambda*sum(s_shrunk);
        if k == 1
            info.relChange(k) = Inf;
        else
            info.relChange(k) = norm(X - X_old,'fro') / max(1e-12, norm(X_old,'fro'));
        end

        if verbose
            fprintf('Iter %3d: obj = %.4e, rank = %d, relChange = %.2e\n', ...
                    k, info.obj(k), r, info.relChange(k));
        end

        % Stopping criterion
        if info.relChange(k) < tol
            info.obj       = info.obj(1:k);
            info.relChange = info.relChange(1:k);
            info.rank      = info.rank(1:k);
            break;
        end
    end
    info.iters = find(info.obj ~= 0, 1, 'last');
    if isempty(info.iters)
        info.iters = 0;
    end

end

