close all
clear all
clc

% Simulation time comparison data
num_train_data = 800;
traditional_sim = 472;      % Traditional simulation time (seconds)
offline_data_gen = traditional_sim * num_train_data;  % Time to generate 800 samples (seconds)
offline_training = 1500;   % Neural operator training time (seconds)
online_inference = 0.01;    % Inference time per simulation (seconds)


% Create breakeven analysis plot
figure;
N = 0:2000;  % Number of simulations
ml_total = (offline_data_gen + offline_training) + N*online_inference;
traditional_total = N*traditional_sim;

% Create plot with increased font size
semilogy(N, ml_total, 'b-', N, traditional_total, 'r--', 'LineWidth', 3);
ax = gca;
ax.FontSize = 12;  % Set axes font size

xlabel('Number of Simulations', 'FontSize', 12);
ylabel('Log-Scale Total Time (seconds)', 'FontSize', 12);
legend('ML Solver Total Time', 'Traditional Solver Total Time',...
       'Location', 'southeast', 'FontSize', 12);
grid on;

% Find breakeven point and add annotation
breakeven = floor((offline_data_gen + offline_training) / (traditional_sim - online_inference)) + 1; 
annotation('textbox', [0.47,0.73,0.3,0.1],...
           'String', sprintf('Breakeven at %d simulations', breakeven),...
           'FitBoxToText', 'on',...
           'FontSize', 12);