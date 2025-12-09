function results = run_nuclear_norm_function(A, lambdas)

    nA = norm(A,'fro');
    nL = numel(lambdas);
    
    errors      = zeros(nL,1);
    ranks       = zeros(nL,1);
    runtimes    = zeros(nL,1);
    iterations  = zeros(nL,1);
    solutions   = cell(nL,1);
    infos       = cell(nL,1); 

    

    for i = 1:nL
        lambda = lambdas(i);
        fprintf('=== lambda = %.4f ===\n', lambda);
    
        tStart = tic;
        [X_lambda, info] = nuclear_norm_svt(A, lambda, ...
                                            'MaxIter', 300, ...
                                            'Tol', 1e-4, ...
                                            'Verbose', false);
        runtimes(i)   = toc(tStart);
        solutions{i}  = X_lambda;
        infos{i}      = info;
    
        % Compute effective rank via SVD of the solution
        [U,S,V] = svd(X_lambda, 'econ');
        s = diag(S);
        r_eff = nnz(s > 1e-6);    % threshold can be tuned
        ranks(i) = r_eff;
    
        % Final relative Frobenius error
        errors(i) = norm(A - X_lambda, 'fro') / nA;
    
        % Iterations
        if isfield(info, 'iters')
            iterations(i) = info.iters;
        else
            iterations(i) = numel(info.obj);
        end
    
        fprintf('  rank ~ %d, rel error = %.4f, iters = %d, time = %.3fs\n', ...
                r_eff, errors(i), iterations(i), runtimes(i));
    end

    results.lambdas   = lambdas;
    results.errors    = errors;
    results.ranks     = ranks;
    results.runtimes  = runtimes;
    results.iterations = iterations;
    results.solutions = solutions;
    results.infos      = infos;

end
