function [] = plotsetting(y_label, subfig_title, axes_size, title_size, legend_size)

if nargin<3
  axes_size = 9; title_size = 11; legend_size= 10;
end


set(gca,'FontName','Arial','FontSize',axes_size,'FontWeight','Bold');
%AX = legend('Proposed', 'IAT', 'NS');
%set(AX,'Location','southeast','FontName','Arial','FontSize',legend_size, 'FontWeight','Bold');
xlabel ('Frequency (kHz)');
ylabel (y_label);
title(subfig_title,'FontSize', title_size, 'FontWeight', 'bold');
% xticks([-10 -5 0 5 10])
% xticklabels({'-10','-5','0','5','10'})

end
