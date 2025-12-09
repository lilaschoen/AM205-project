function [X, info] = image_recomposition_nuclear_norm_function(A_obs, M, lambda, X_init, varargin)
    A_true = [];
    maxIter = 200;
    tol     = 1e-4;
    tau     = 1.0;
    verbose = false;

    for k = 1:2:length(varargin)
        name = lower(varargin{k});
        val  = varargin{k+1};
        switch name
            case 'maxiter',  maxIter = val;
            case 'tol',      tol     = val;
            case 'stepsize', tau     = val;
            case 'verbose',  verbose = val;
            case 'a_true', A_true = val;
        end
    end

    if isempty(X_init)
        X = zeros(size(A_obs));
    else
        X = X_init;
    end

    info.obj       = zeros(maxIter,1);
    info.relChange = zeros(maxIter,1);
    info.rank      = zeros(maxIter,1);
    if ~isempty(A_true)
        info.relErr_full = zeros(maxIter,1);
        normA = norm(A_true,'fro');
    else
        info.relErr_full = [];
    end

    for it = 1:maxIter

        X_old = X;

        % gradient
        G = (X - A_obs) .* M;

        % gradient descent step
        Y = X - tau * G;

        % SVT step
        [U,S,V] = svd(Y,'econ');
        s = diag(S);
        s_shrunk = max(s - tau*lambda, 0);

        r = nnz(s_shrunk > 0);
        if r == 0
            X = zeros(size(X));
        else
            X = U(:,1:r) * diag(s_shrunk(1:r)) * V(:,1:r)';
        end

        info.rank(it) = r;
        info.obj(it)  = 0.5*norm(G,'fro')^2 + lambda*sum(s_shrunk);

        relChange = norm(X - X_old,'fro') / max(1e-12, norm(X_old,'fro'));
        info.relChange(it) = relChange;

        if ~isempty(A_true)
            info.relErr_full(it) = norm(X - A_true,'fro') / normA;
        end

        if verbose
            fprintf('Iter %3d: obj=%.4e, rank=%3d, rel=%.2e\n', ...
                    it, info.obj(it), r, relChange);
        end

        if relChange < tol
            % Trim histories to actual number of iterations
            info.obj       = info.obj(1:it);
            info.relChange = info.relChange(1:it);
            info.rank      = info.rank(1:it);
            info.iters     = it;

            fprintf('SVT converged at iter %d (relChange = %.2e)\n', it, relChange);
            return;
        end
    end

    info.obj       = info.obj(1:maxIter);
    info.relChange = info.relChange(1:maxIter);
    info.rank      = info.rank(1:maxIter);
    info.iters     = maxIter;
    info.relErr_full = info.relErr_full(1:it);

    fprintf('SVT reached maxIter = %d without hitting tol (final relChange = %.2e)\n', ...
            maxIter, info.relChange(end));
end