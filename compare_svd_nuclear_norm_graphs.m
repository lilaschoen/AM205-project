%% Define images and lambda grid
image_files  = {'lincoln.png','nyc.jpg','fields.jpg'};
image_labels = { 'Lincoln','NYC skyline', 'Field'};

numImages = numel(image_files);

lambdas = [0.005 0.01 0.02 0.05 0.1 0.2 0.4 1 2 5 10];

rank5_all     = cell(numImages,1);
svd_err5_all  = cell(numImages,1);
nuc_err5_all  = cell(numImages,1);
svd_time5_all = cell(numImages,1);
nuc_time5_all = cell(numImages,1);
A_all         = cell(numImages,1);

%%  Perfomr nuclear norm sweep, pick 5 smallest ranks and run SVD
for i = 1:numImages

    % Load & normalize
    A = imread(image_files{i});
    A = im2gray(A);
    A = im2double(A);
    A_all{i} = A;


    % Nuclear norm sweep
    svt = run_nuclear_norm_function(A, lambdas);
    all_ranks = svt.ranks(:);

    % Select 5 smallest ranks
    [~, sort_idx] = sort(all_ranks, 'ascend');
    k = min(5, numel(all_ranks));
    keep_idx = sort_idx(1:k);

    lambda5   = lambdas(keep_idx);
    rank5     = all_ranks(keep_idx);
    nuc_err5  = svt.errors(keep_idx);
    nuc_time5 = svt.runtimes(keep_idx);

    % Run truncated SVD at these ranks
    tsvd5 = run_truncated_svd_function(A, rank5);
    svd_err5  = tsvd5.errors(:);
    svd_time5 = tsvd5.runtimes(:);

    rank5_all{i}     = rank5(:);
    svd_err5_all{i}  = svd_err5(:);
    nuc_err5_all{i}  = nuc_err5(:);
    svd_time5_all{i} = svd_time5(:);
    nuc_time5_all{i} = nuc_time5(:);
end

%% Plot Error vs Normalized Rank and Error vs Runtime

colors = lines(numImages);

figure('Position',[100 100 1200 450]);

% Error vs Normalized Rank
subplot(1,2,1); hold on;

for i = 1:numImages
    A = A_all{i};
    [m,n] = size(A);

    r_norm = rank5_all{i} / min(m,n);
    e_s    = svd_err5_all{i};
    e_n    = nuc_err5_all{i};

    semilogy(r_norm, e_s, '-o', ...
        'Color', colors(i,:), 'LineWidth', 2, ...
        'DisplayName', [image_labels{i} ' - SVD']);

    semilogy(r_norm, e_n, '--s', ...
        'Color', colors(i,:), 'LineWidth', 2, ...
        'DisplayName', [image_labels{i} ' - Nuclear']);
end

hold off;
xlabel('Normalized rank r / min(m,n)', 'FontSize', 14);
ylabel('Relative Error', 'FontSize', 14);
title('Relative Frobenius Error vs Normalized Rank', 'FontSize', 16);
grid on;
legend('Location','northeastoutside', 'FontSize', 11);
set(gca,'FontSize',12);

% Error vs Runtime
subplot(1,2,2); hold on;

for i = 1:numImages
    t_s = svd_time5_all{i};
    e_s = svd_err5_all{i};

    t_n = nuc_time5_all{i};
    e_n = nuc_err5_all{i};

    plot(t_s, e_s, '-o', ...
        'Color', colors(i,:), 'LineWidth', 2, ...
        'DisplayName', [image_labels{i} ' - SVD']);

    plot(t_n, e_n, '--s', ...
        'Color', colors(i,:), 'LineWidth', 2, ...
        'DisplayName', [image_labels{i} ' - Nuclear']);
end

hold off;
xlim([0, 3]);
xlabel('Runtime (seconds)', 'FontSize', 14);
ylabel('Relative Error', 'FontSize', 14);
title('Relative Frobenius Error vs Runtime', 'FontSize', 16);
grid on;
legend('Location','northeastoutside', 'FontSize', 11);
set(gca,'FontSize',12);