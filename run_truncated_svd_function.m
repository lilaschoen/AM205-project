function results = run_truncated_svd_function(A, r_values)

    nA = norm(A,'fro');
    [U,S,V] = svd(A,'econ');
    s = diag(S);

    nR = length(r_values);

    errors     = zeros(nR,1); 
    ranks      = zeros(nR,1); 
    runtimes   = zeros(nR,1);
    solutions  = cell(nR,1);

    for i = 1:nR
        r = r_values(i);
        ranks(i) = r;
    
        tStart = tic;
    
        % rank-r truncated SVD reconstruction
        Ur = U(:,1:r);
        Sr = diag(s(1:r));
        Vr = V(:,1:r);
    
        Ar = Ur * Sr * Vr';
    
        runtimes(i) = toc(tStart);
        solutions{i} = Ar;
    
        % Frobenius errors
        errors(i) = norm(A - Ar,'fro') / nA;
    
        fprintf('  rank = %3d | rel error = %.4f | time = %.4f s\n', ...
                r, errors(i), runtimes(i));
    end

    results.r_values  = r_values;
    results.errors    = errors;
    results.ranks     = ranks;
    results.runtimes  = runtimes;
    results.solutions = solutions;

end
