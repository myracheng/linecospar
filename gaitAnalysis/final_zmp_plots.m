%%%%%%%%%%%%%%%%%%%  Final ZMP plots for IROS 2020 %%%%%%%%%%%%%%%%%%%%%%%%
%%% This script plots the CoM and CoP of the subjects 1 and 5 for the gaits
%%% corresponding to their most and least preferred gaits


%% Load Information from Validation Trials
% load the precompiled structure of ZMP variables
% the CoP and CoM were calculated from the experimental exoskeleton data
if ~exist('patientLogData')
    load('patientValLogData.mat')
end

%indices of validation trials corresponding to a_max for each subject
goodTrials = repmat(4,6,1);

%indices of validation trial corresponding to min value of posterior mean
%for each subject
badTrials = [3; ...
    3; ...
    5; ...
    1;...
    3; ...
    3];


%% Plot CoM and CoP for many steps (animated plotting)
for p = [1,5] %only evaluate subject 1 and 5
    
    % create a new figure for each subject
    f1 = figure();
    suptitle(sprintf('Subject %i',p))
    
    % cycle through all the validation trials looking for the trials
    % corresponding to the highest and lowest value on the posterior mean
    for num = 1:8
        
        % if the trial is the validation trial correponding to the
        % minimum posterior mean value then plot its ZMP
        if num == badTrials(p)
            
            % extract the data from the patientLogData structure
            log = patientLogData(p,num);
            
            % plot the CoM and CoP
            fprintf('Plotting ZMP for Subject %i, Validation %i . . . \n',p,num);
            ax = subplot(2,1,1);
            PlottingTools.plotZMP_animated(log,ax)
            
            % format subplot
            ylabel(ax,{'Position in';'y-direction (m)'})
            title(ax,'Not Preferred')
            axis(ax,'equal')
            xlim(ax,[0,1])
            ylim(ax,[-0.2,0.5])
            hLegend = findall(gcf,'tag','legend');
            hLegend(1).Position = [0.295333055369006,0.48485347486169,0.538584937487059,0.060678924226019];
            
            % if the trial is the validation trial correponding to the
            % minimum posterior mean value then plot its ZMP
        elseif num == goodTrials(p)
            
            % create a new subplot
            ax = subplot(2,1,2);
            
            % format the size of the subplot
            axis(ax,'equal')
            xlim(ax,[0,1])
            ylim(ax,[-0.2,0.5])
            
            % plot the CoM and CoP
            fprintf('Plotting ZMP for Subject %i, Validation %i . . . \n',p,num)
            PlottingTools.plotZMP_animated(log,ax)
            
            % format subplot
            ylabel(ax,{'Position in';'y-direction (m)'})
            xlabel(ax,'Position in x-direction (m)')
            title(ax,{'';'';'';'';'';'Preferred'})
            hLegend = findall(gcf,'tag','legend');
            hLegend(1).Visible = 'off';
            
            % change the font to latex formatting
            PlottingTools.latexify;
            
            % change the font to size 16
            PlottingTools.fontsize(16);
            
            % change the figure size
            f1.Position =  [743 1 613 968];
            f1.PaperPositionMode = 'auto';
            f1.PaperSize = [29.72/2,21.0];
        end
    end
end

%% Plot ZMP variables for a single gait cycle
for p = [1,5] %only evaluate subject 1 and 4
    
    % create a new figure for each subject
    f1 = figure();
    
    % cycle through all the validation trials looking for the trials
    % corresponding to the highest and lowest value on the posterior mean
    for num = 1:8
        
        % extract the data from the patientLogData structure
        log = patientLogData(p,num);
        
        % if the trial is the validation trial correponding to the
        % minimum posterior mean value then plot its ZMP
        if num == badTrials(p)
            
            % plot ZMP variables for one gait cycle
            fprintf('Plotting ZMP of Subject %i, Validation %i . . . \n',p,num);
            ax = subplot(2,1,1);
            ph = PlottingTools.plotZMP_single(log,ax);
            
            % format subplot
            title('Least Preferred')
            ylabel('y-axis (m)')
            xlabel({'x-axis (m)';'';''})
            axis equal
            xlim([-0.1,0.7])
            ylim([-0.2,0.5])
            
        elseif num == goodTrials(p)
            fprintf('Plotting ZMP of Subject %i, Validation %i . . . \n',p,num);
            ax = subplot(2,1,2);
            ph = PlottingTools.plotZMP_single(log,ax);
            
            % format subplot            
            title('Most Preferred')
            ylabel('y-axis (m)')
            xlabel({'x-axis (m)';'';''})
            axis equal
            xlim([-0.1,0.7])
            ylim([-0.2,0.5])
            
        end
    end
    
    % add legend to plots
    legend([ph(1),ph(2),ph(3),ph(4),ph(5)],{'$x_{\textrm{COM}}$','$x^{\textrm{goal}}_{\textrm{COM}}$','$p^{\textrm{goal}}$','$p$','$x_{\textrm{COP}}$'},'Orientation','horizontal','Position',[0.1111    0.4996    0.8251    0.0425])
    
    % make font formatting latex
    PlottingTools.latexify;
    
    % make font size 16
    PlottingTools.fontsize(16);
    
    % format size of figure
    f1.Position = [680 4 580 965];
    f1.PaperPositionMode = 'auto';
    f1.PaperSize = [29.72/2,21.0];
    
end