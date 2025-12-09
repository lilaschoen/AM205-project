function [U, V, info] = image_recomposition_low_rank_factorization_function( ...
    A_obs, M, U_init, V_init, r, maxIter, verbose, varargin)
% Alternating Least Squares (ALS) for low-rank matrix factorization
    [m,n] = size(A_obs);

    A_true = [];
    for k = 1:2:length(varargin)
        name = lower(varargin{k});
        val  = varargin{k+1};
        switch name
            case 'a_true'
                A_true = val;
        end
    end

    % initialize U,V
    if isempty(U_init)
        U = randn(m, r);
    else
        U = U_init;
    end
    if isempty(V_init)
        V = randn(n, r);
    else
        V = V_init;
    end

    info.maskedError = zeros(maxIter,1); 
    if ~isempty(A_true)
        info.relErr_full = zeros(maxIter,1); 
        normA = norm(A_true, 'fro');
    else
        info.relErr_full = [];
        normA = NaN;
    end

    for it = 1:maxIter

        % update V (fix U)
        for j = 1:n
            mask_col = M(:,j);
            if any(mask_col)
                U_sub = U(mask_col,:);
                A_sub = A_obs(mask_col,j);
                V(j,:) = U_sub \ A_sub;
            end
        end

        % update U (fix V)
        for i = 1:m
            mask_row = M(i,:);
            if any(mask_row)
                V_sub = V(mask_row,:);
                A_sub = A_obs(i,mask_row)';
                U(i,:) = V_sub \ A_sub;
            end
        end

        X = U * V';

        % masked error
        masked_err = norm((X - A_obs).*M, 'fro');
        info.maskedError(it) = masked_err;

        if ~isempty(A_true)
            info.relErr_full(it) = norm(X - A_true, 'fro') / normA;
        end

        if verbose
            if ~isempty(A_true)
                fprintf('ALS iter %d | masked=%.4f | relErr_full=%.4f\n', ...
                        it, masked_err, info.relErr_full(it));
            else
                fprintf('ALS iter %d | masked=%.4f\n', it, masked_err);
            end
        end
    end
end