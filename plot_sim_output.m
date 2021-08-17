best_score = score_list(end);
best_input = input_list(end,:); %these should be checked against graph

best_plot = round((best_input./size(runInfo.traj_xX,1)).*length(runInfo.t),0);

total_vel = sqrt(runInfo.vX.^2+runInfo.vY.^2);

infotitle = ['Run Info: Te = ', num2str(round(runInfo.Te,3)), ', Blur Ratio = ', num2str(runInfo.blur_ratio), ', Number of Images = ', num2str(runInfo.num_images)];

figure();
sgtitle(infotitle); 

subplot(3, 2, 1);
plot(runInfo.t, runInfo.xX, '-k');
hold on
plot(runInfo.t(best_plot), runInfo.xX(best_plot), 'or');
title('Horizontal Position and Selected Captures');
xlabel('Time [s]');
ylabel('Position [m]');

subplot(3, 2, 2);
plot(runInfo.t, runInfo.xY, '-k');
hold on
plot(runInfo.t(best_plot), runInfo.xY(best_plot), 'or');
title('Vertical Position and Selected Captures');
xlabel('Time [s]');
ylabel('Position [m]');

subplot(3, 2, 3);
plot(runInfo.t, runInfo.vX, '-k');
hold on
plot(runInfo.t(best_plot), runInfo.vX(best_plot), 'or');
title('Horizontal Velocity and Selected Captures');
xlabel('Time [s]');
ylabel('Velocity [m/s]');

subplot(3, 2, 4);
plot(runInfo.t, runInfo.vY, '-k');
hold on
plot(runInfo.t(best_plot), runInfo.vY(best_plot), 'or');
title('Vertical Velocity and Selected Captures');
xlabel('Time [s]');
ylabel('Velocity [m/s]');

subplot(3, 2, 5);
plot(runInfo.t, total_vel, '-k');
hold on
plot(runInfo.t(best_plot), total_vel(best_plot), 'or');
title('Total Velocity and Selected Captures');
xlabel('Time [s]');
ylabel('Velocity [m/s]');

subplot(3, 2, 6);
plot(1:1:length(score_list), score_list, '-b');
hold on
title('Optimization Score');
xlabel('Iteration');
ylabel('Score');