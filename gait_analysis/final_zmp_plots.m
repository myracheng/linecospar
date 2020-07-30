%% Final ZMP plots for IROS 2020

% load the precompiled structure of ZMP variables
if ~exist('patientLogData')
    load('patientValLogData.mat')
end

%validation trials corresponding to a_max
goodTrials = repmat(4,6,1);

%validation trial corresponding to minimum value of posterior mean
badTrials = [3; ...
    3; ...
    5; ...
    1;...
    3; ...
    3];


%% Plot ZMP variables
for p = [1,5] %only evaluate patients 1 and 5
    f1 = figure();
    suptitle(sprintf('Subject %i',p))
    
    % cycle through all the validation trials looking for the trials
    % corresponding to the highest and lowest value on the posterior mean
    for num = 1:8
        
        % extract the data from the patientLogData structure
        log = patientLogData(p,num);
        numsteps =    patientLogData(p,num).numsteps;
        newStepInds = patientLogData(p,num).newStepInds;
        ff_q = patientLogData(p,num).ff_q;
        stanceFoot = patientLogData(p,num).stanceFoot;
        swingFoot = patientLogData(p,num).swingFoot;
        com = patientLogData(p,num).com;
        com_dot = patientLogData(p,num).com_dot;
        com_ddot = patientLogData(p,num).com_ddot;
        cop = patientLogData(p,num).cop;
        cop_dot = patientLogData(p,num).cop_dot;
        cop_ddot = 0;
        p_dot = 0;
        p_ddot = 0;
        
        % if the trial is the validation trial correponding to the
        % minimum posterior mean value then plot its ZMP
        if num == badTrials(p)
            ax = subplot(2,1,1);
            
            fprintf('Plotting ZMP for Subject %i, Validation %i . . . \n',p,num);
            PlottingTools.plotZMP(stanceFoot,newStepInds,com,cop,log,3,ax)
            
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
            ax = subplot(2,1,2);
            
            axis(ax,'equal')
            xlim(ax,[0,1])
            ylim(ax,[-0.2,0.5])
            
            fprintf('Plotting ZMP for Subject %i, Validation %i . . . \n',p,num)
            PlottingTools.plotZMP(stanceFoot,newStepInds,com,cop,log,3,ax)
            
            ylabel(ax,{'Position in';'y-direction (m)'})
            xlabel(ax,'Position in x-direction (m)')
            title(ax,{'';'';'';'';'';'Preferred'})
            hLegend = findall(gcf,'tag','legend');
            hLegend(1).Visible = 'off';
            
            PlottingTools.latexify;
            PlottingTools.fontsize(16);
            f1.Position =  [743 1 613 968];
            f1.PaperPositionMode = 'auto';
            f1.PaperSize = [29.72/2,21.0];
            %                 figName = ['Figures/',sprintf('Subject_%i_gaitTiles.pdf',p)];
            %                 print(f1,figName,'-dpdf','-r300');
            %                 system(['pdfcrop ', figName,' ',figName]);
            %                 close all;
        end
    end
end

%% Plot ZMP
colorblind = [55,126,184; ...
    255, 127, 0;...
    77, 175, 74;...
    247, 129, 191;...
    166, 86, 40;...
    152, 78, 163;...
    153, 153, 153;...
    228, 26, 28;...
    222, 222, 0]/255;

load([pwd,'/+PlottingTools/params.mat'])

for p = [1,5]
    f1 = figure();
    for num = 1:8
        
        if isempty(patientLogData(p,num).t)
            emptylog = 1;
        else
            emptylog = 0;
        end
        
        log = patientLogData(p,num);
        
        if ~emptylog
            numsteps =    patientLogData(p,num).numsteps;
            newStepInds = patientLogData(p,num).newStepInds;
            ff_q = patientLogData(p,num).ff_q;
            stanceFoot = patientLogData(p,num).stanceFoot;
            swingFoot = patientLogData(p,num).swingFoot;
            com = patientLogData(p,num).com;
            com_dot = patientLogData(p,num).com_dot;
            com_ddot = patientLogData(p,num).com_ddot;
            cop = patientLogData(p,num).cop;
            cop_dot = patientLogData(p,num).cop_dot;
            %             cop_ddot = patientLogData(p,num).cop_ddot;
            %             p_dot = patientLogData(p,num).p_dot;
            %             p_ddot = patientLogData(p,num).p_ddot;
            cop_ddot = 0;
            p_dot = 0;
            p_ddot = 0;
            
            if num == badTrials(p)
                
                fprintf('Plotting ZMP of Subject %i, Validation %i . . . \n',p,num);
                subplot(2,1,1);
                %firststep
                stepInds = newStepInds(2):newStepInds(3);
                COM_goal = repmat((stanceFoot([1,2],stepInds(end))+stanceFoot([1,2],stepInds(1)))/2,1,length(stepInds));
                COM_goal(1,:) = COM_goal(1,:) + 0.1794;
                COM_step = com([1,2],stepInds);
                COP_step = cop([1,2],stepInds);
                Pos_ref= repmat(stanceFoot([1,2],stepInds(end)),1,length(stepInds(1:end-1)));
                Pos = swingFoot([1,2],stepInds(1:end-1));
                
                plot(COM_step(1,:),COM_step(2,:),'--','color',[colorblind(1,:),0.5])
                
                
                
                [x, y, w, h] = soleToRect([stanceFoot(:,[stepInds(1),stepInds(end)+1]),swingFoot(:,stepInds(1))],3,log,params);
                hold on
                for fi = 1:3
                    foot = rectangle('Position',[x(fi),y(fi),w(fi),h(fi)],'EdgeColor',[colorblind(7,:),0.5]);
                end
                plot(COM_goal(1,:),COM_goal(2,:),'p','MarkerSize',10,'color',[colorblind(1,:),0.5])
                plot(Pos_ref(1,:),Pos_ref(2,:),'p','MarkerSize',10,'color',[colorblind(3,:),0.5])
                plot(Pos(1,:),Pos(2,:),'--','color',[colorblind(3,:),0.5])
                plot(COP_step(1,:),COP_step(2,:),'--','color',[colorblind(2,:),0.5])
                
                %second step
                stepInds = newStepInds(3):newStepInds(4);
                COM_goal = repmat((stanceFoot([1,2],stepInds(end))+stanceFoot([1,2],stepInds(1)))/2,1,length(stepInds));
                COM_goal(1,:) = COM_goal(1,:) + 0.1794;
                COM_step = com([1,2],stepInds);
                COM_vel = com_dot([1,2],stepInds);
                COM_accel = com_ddot([1,2],stepInds);
                COP_step = cop([1,2],stepInds);
                COP_vel = cop_dot([1,2],stepInds);
                Pos_ref = repmat(stanceFoot([1,2],stepInds(end)),1,length(stepInds(1:end-1)));
                Pos = swingFoot([1,2],stepInds(1:end-1));
                
                p1 = plot(COM_step(1,:),COM_step(2,:),'color',colorblind(1,:));
                [x, y, w, h] = soleToRect([stanceFoot(:,[stepInds(1),stepInds(end)+1]),swingFoot(:,stepInds(1))],3,log,params);
                hold on
                for fi = 1:3
                    foot = rectangle('Position',[x(fi),y(fi),w(fi),h(fi)],'EdgeColor',colorblind(7,:));
                end
                p2 = plot(COM_goal(1,:),COM_goal(2,:),'p','MarkerSize',10,'color',colorblind(1,:));
                p3 = plot(Pos_ref(1,:),Pos_ref(2,:),'p','MarkerSize',10,'color',colorblind(3,:));
                p4 = plot(Pos(1,:),Pos(2,:),'.','color',colorblind(3,:));
                p5 = plot(COP_step(1,:),COP_step(2,:),'color',colorblind(2,:));
                
            elseif num == goodTrials(p)
                fprintf('Plotting ZMP of Subject %i, Validation %i . . . \n',p,num);
                subplot(2,1,2);
                %firststep
                stepInds = newStepInds(2):newStepInds(3);
                
                COM_goal = repmat((stanceFoot([1,2],stepInds(end))+stanceFoot([1,2],stepInds(1)))/2,1,length(stepInds));
                COM_goal(1,:) = COM_goal(1,:) + 0.1794;
                COM_step = com([1,2],stepInds);
                COP_step = cop([1,2],stepInds);
                Pos_ref= repmat(stanceFoot([1,2],stepInds(end)),1,length(stepInds(1:end-1)));
                Pos = swingFoot([1,2],stepInds(1:end-1));
                
                plot(COM_step(1,:),COM_step(2,:),'--','color',[colorblind(1,:),0.5])
                [x, y, w, h] = soleToRect([stanceFoot(:,[stepInds(1),stepInds(end)+1]),swingFoot(:,stepInds(1))],3,log,params);
                hold on
                for fi = 1:3
                    foot = rectangle('Position',[x(fi),y(fi),w(fi),h(fi)],'EdgeColor',[colorblind(7,:),0.5]);
                end
                plot(COM_goal(1,:),COM_goal(2,:),'p','MarkerSize',10,'color',[colorblind(1,:),0.5])
                plot(Pos_ref(1,:),Pos_ref(2,:),'p','MarkerSize',10,'color',[colorblind(3,:),0.5])
                plot(Pos(1,:),Pos(2,:),'--','color',[colorblind(3,:),0.5])
                plot(COP_step(1,:),COP_step(2,:),'--','color',[colorblind(2,:),0.5])
                
                %second step
                stepInds = newStepInds(3):newStepInds(4);
                
                COM_goal = repmat((stanceFoot([1,2],stepInds(end))+stanceFoot([1,2],stepInds(1)))/2,1,length(stepInds));
                COM_goal(1,:) = COM_goal(1,:) + 0.1794;
                COM_step = com([1,2],stepInds);
                COM_vel = com_dot([1,2],stepInds);
                COM_accel = com_ddot([1,2],stepInds);
                COP_step = cop([1,2],stepInds);
                COP_vel = cop_dot([1,2],stepInds);
                Pos_ref = repmat(stanceFoot([1,2],stepInds(end)),1,length(stepInds(1:end-1)));
                Pos = swingFoot([1,2],stepInds(1:end-1));
                
                p1 = plot(COM_step(1,:),COM_step(2,:),'color',colorblind(1,:));
                [x, y, w, h] = soleToRect([stanceFoot(:,[stepInds(1),stepInds(end)+1]),swingFoot(:,stepInds(1))],3,log,params);
                hold on
                for fi = 1:3
                    foot = rectangle('Position',[x(fi),y(fi),w(fi),h(fi)],'EdgeColor',colorblind(7,:));
                end
                p2 = plot(COM_goal(1,:),COM_goal(2,:),'p','MarkerSize',10,'color',colorblind(1,:));
                p3 = plot(Pos_ref(1,:),Pos_ref(2,:),'p','MarkerSize',10,'color',colorblind(3,:));
                p4 = plot(Pos(1,:),Pos(2,:),'.','color',colorblind(3,:));
                p5 = plot(COP_step(1,:),COP_step(2,:),'color',colorblind(2,:));
                
            end
        end
    end
    subplot(2,1,1)
    title('Least Preferred')
    ylabel('y-axis (m)')
    xlabel({'x-axis (m)';'';''})
    axis equal
    xlim([-0.1,0.7])
    ylim([-0.2,0.5])
    % xticks([0,0.2,0.4,0.6])
    
    subplot(2,1,2)
    title('Most Preferred')
    ylabel('y-axis (m)')
    xlabel({'x-axis (m)';'';''})
    axis equal
    xlim([-0.1,0.7])
    ylim([-0.2,0.5])
    % xticks([0,0.2,0.4,0.6])
    
    legend([p1(1),p2(1),p3(1),p4(1),p5(1)],{'$x_{\textrm{COM}}$','$x^{\textrm{goal}}_{\textrm{COM}}$','$p^{\textrm{goal}}$','$p$','$x_{\textrm{COP}}$'},'Orientation','horizontal','Position',[0.1111    0.4996    0.8251    0.0425])
    
    PlottingTools.latexify;
    PlottingTools.fontsize(16);
    
    f1.Position = [680 4 580 965];
    f1.PaperPositionMode = 'auto';
    f1.PaperSize = [29.72/2,21.0];
    %     figName = ['Figures/',sprintf('Subject_%i_ZMPgaitTiles.pdf',p)];
    %     print(f1,figName,'-dpdf','-r300');
    %     system(['pdfcrop ', figName,' ',figName]);
    
end



%% helper functions
function [x, y, w, h] = soleToRect(sole,dim,log,params)

solexy = [sole(1,:);sole(2,:)];

tl = solexy+[params.rFoot.lengthToToe;params.rFoot.width];
tr = solexy+[params.rFoot.lengthToToe;-params.rFoot.width];
bl = solexy+[-params.rFoot.lengthToHeel;params.rFoot.width];
br = solexy+[-params.rFoot.lengthToHeel;-params.rFoot.width];

if dim == 3
    x = br(1,:);
elseif dim == 2
    x =  log.t;
end
y = br(2,:);
w = sum([params.rFoot.lengthToToe,params.rFoot.lengthToToe])*ones(1,size(x,2));
h = 2*params.rFoot.width*ones(1,size(x,2));
end

function ff_q = getFreeFlyer(log,stanceFoot,newStanceInds,q)
stance = log.customFields(3,:) + 1;
tempstance = stance;
stance(tempstance == 1) = 2;
stance(tempstance == 2) = 1;

% q = [q(7:12,:);q(1:6,:)];
ff_q = zeros(6,size(stanceFoot,2));
for t = 1:size(stanceFoot,2)
    if stance(t) == 1
        ffFrame = -h_LeftSoleJoint_LeftSS([zeros(6,1);q(:,t)], zeros(6,1));
    else
        ffFrame = -h_RightSoleJoint_RightSS([zeros(6,1);q(:,t)], zeros(6,1));
    end
    ff_q(:,t) = stanceFoot(:,t)-ffFrame;
    %     ff_q(1,t) = -ff_q(1,t);
end
%     ff_q = -ff_q;
end