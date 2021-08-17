function [] = plot_single_trial(ind, runInfo, all_input, all_score, all_t, all_xX, all_vX, all_xY, all_vY)

input = all_input(:,ind);
score = all_score(:,ind);

t = all_t(:,ind);
t = t - t(1);
xX = all_xX(:,ind);
vX = all_vX(:,ind);
xY = all_xY(:,ind);
vY = all_vY(:,ind);

total_vel = sqrt(vX.^2+vY.^2);

num_indexes = floor(runInfo.secondsRange./runInfo.Te);
indexes = round(input./num_indexes.*length(t),0);

single = round(15./num_indexes.*length(t),0);
triple = round([5,15,25]./num_indexes.*length(t),0);

% infotitle = ['Run Info: Te = ', num2str(round(runInfo.Te,3)), ', Blur Ratio = ', num2str(runInfo.blur_ratio), ', Number of Images = ', num2str(runInfo.num_images)];

figure();
font = 11;
tiledlayout(2,1, 'TileSpacing', 'compact')
ax(1) = nexttile;

% sgtitle(infotitle); 

% subplot(3, 2, 1);
% plot(t, xX, '-k');
% hold on
% plot(t(single), xX(single), 'or', t(triple), xX(triple), 'sg', t(indexes), xX(indexes), 'xb');
% title('Horizontal Position and Selected Captures');
% xlabel('Time [s]');
% ylabel('Position [m]');
% 
% subplot(3, 2, 2);
% plot(t, xY, '-k');
% hold on
% plot(t(single), xY(single), 'or', t(triple), xY(triple), 'sg', t(indexes), xY(indexes), 'xb');
% title('Vertical Position and Selected Captures');
% xlabel('Time [s]');
% ylabel('Position [m]');


plot(t, vX, '-k');
hold on
plot(t(single), vX(single), '.r', t(triple), vX(triple), 'sg', t(indexes), vX(indexes), 'xb', 'MarkerSize',14, 'LineWidth',2);
title('Horizontal Velocity and Selected Captures','FontSize', font);

xlabel('Time [s]','FontSize', font);
set(gca,'FontSize',font);
ylabel('Velocity [m/s]','FontSize', font);
hold off

ax(2) = nexttile;
plot(t, vY, '-k');
hold on
plot(t(single), vY(single), '.r', t(triple), vY(triple), 'sg', t(indexes), vY(indexes), 'xb', 'MarkerSize',14, 'LineWidth',2);

title('Vertical Velocity and Selected Captures','FontSize', font);
xlabel('Time [s]','FontSize', font);
ylabel('Velocity [m/s]','FontSize', font);
set(gca,'FontSize',font);
lh =legend(ax(2),'','Single Capture','Equally Spaced 3','Optimized 3','Location','NorthOutside','Orientation','Horizontal', 'FontSize', 12);
lh.Layout.Tile = 'South'; % <----- relative to tiledlayout

% subplot(3, 2, 3);
% plot(t, total_vel, '-k');
% hold on
% plot(t(single), total_vel(single), 'or', t(triple), total_vel(triple), 'sg', t(indexes), total_vel(indexes), 'xb');
% title('Total Velocity and Selected Captures');
% xlabel('Time [s]');
% ylabel('Velocity [m/s]');

end