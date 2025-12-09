% Lincoln
A1 = imread('lincoln.png');
A1 = im2gray(A1);
A1 = double(A1) / 255;

% NYC skyline
A2 = imread('nyc.jpg');
A2 = im2gray(A2);
A2 = double(A2) / 255;

% Field
A3 = imread('fields.jpg');
A3 = im2gray(A3);
A3 = double(A3) / 255;

% Compute SVDs
[U1,S1,V1] = svd(A1, 'econ'); s_lincoln = diag(S1);
[U5,S5,V5] = svd(A2, 'econ'); s_skyline = diag(S5);
[U6,S6,V6] = svd(A3, 'econ'); s_field = diag(S6);

% Cumulative energies
energy_lincoln = cumsum(s_lincoln.^2) / sum(s_lincoln.^2);
energy_skyline = cumsum(s_skyline.^2) / sum(s_skyline.^2);
energy_field = cumsum(s_field.^2) / sum(s_field.^2);

% Choose rank range
max_r = 100;
max_r = min([max_r, numel(s_lincoln), numel(s_skyline), numel(s_field)]);
r_values = 1:max_r;

% Make wide figure
figure('Position',[100 100 1200 450]);

% Singular value decay (log scale)
subplot(1,2,1);
semilogy(s_lincoln, 'o-', 'DisplayName', 'Lincoln', 'LineWidth', 2.5); hold on;
semilogy(s_skyline, 'x--', 'DisplayName', 'NYC skyline', 'LineWidth', 2.5);
semilogy(s_field, 's--', 'DisplayName', 'Field', 'LineWidth', 2.5);
hold off;
xlabel('Index i', 'FontSize', 14);
ylabel('\sigma_i (log scale)', 'FontSize', 14);
title('Singular value decay', 'FontSize', 16);
legend('show','FontSize',12,'Location','northeast');
set(gca,'FontSize',12);
grid on;

% Cumulative energy vs rank
subplot(1,2,2);
plot(r_values, energy_lincoln(r_values), '-', 'DisplayName', 'Lincoln', 'LineWidth', 2.5); hold on;
plot(r_values, energy_skyline(r_values), '--', 'DisplayName', 'NYC skyline', 'LineWidth', 2.5);
plot(r_values, energy_field(r_values), '--', 'DisplayName', 'Field', 'LineWidth', 2.5);
hold off;
xlabel('Rank r', 'FontSize', 14);
ylabel('Energy captured', 'FontSize', 14);
title('Cumulative energy vs rank', 'FontSize', 16);
legend('show','FontSize',12,'Location','southeast');
set(gca,'FontSize',12);
grid on;