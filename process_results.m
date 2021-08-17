clear; clc;

load('All_Results');

mean_Opti = mean(all_score_Opti);
mean_Triple = mean(all_score_Triple);
mean_Single = mean(all_score_Single);

std_Opti = std(all_score_Opti);
std_Triple = std(all_score_Triple);
std_Single = std(all_score_Single);

[h_Triple, p_Triple] = ttest(all_score_Opti, all_score_Triple, 'Tail', 'left');
[h_Single, p_Single] = ttest(all_score_Opti, all_score_Single, 'Tail', 'left');
