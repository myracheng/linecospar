%% Analyze J_{human}

% load the precompiled structure of ZMP variables
if ~exist('patientLogData')
    load('patientValLogData.mat')
end

goodTrials = repmat([4,7],6,1);
goodTrials(4,:) = [6,8];
badTrials = [1,3,5,8; ...
    3,5,6,8; ...
    1,5,6,8; ...
    1,4,3,5;...
    3,5,6,8; ...
    3,5,6,8];
J_good = zeros(6,8);
J_bad = zeros(6,8);
logInds = 1:8;

for costInd = 1:11
    for p = 1:6
        for num = logInds
            fprintf('Evaluating Subject %i, Validation %i . . . \n',p,num);
            log = patientLogData(p,num);
            if ~isempty(log.t)
                numsteps =    log.numsteps;
                newStepInds = log.newStepInds;
                ff_q = log.ff_q;
                stanceFoot = log.stanceFoot;
                swingFoot = log.swingFoot;
                com = log.com;
                com_dot = log.com_dot;
                com_ddot = log.com_ddot;
                cop = log.cop;
                cop_dot = log.cop_dot;
                cop_ddot = zeros(2,length(cop));
                p_dot = zeros(2,length(cop));
                p_ddot = zeros(2,length(cop));
%                 
                [~,~,~,~,x_5_temp] = ProcessLogs.getErrorsAll(newStepInds,stanceFoot,swingFoot,com,com_dot,com_ddot,cop,cop_dot,cop_ddot,p_dot,p_ddot,log,ff_q);
                
                x_5{num} = x_5_temp{costInd};
                if any(num == goodTrials(p,:))
                    J_good(p,num) = 1*(norm(x_5{num}(1,:),2)) + 1*(norm(x_5{num}(2,:),2)); ...
                        %                             + (norm(x_4{num}(1,:),2)) + norm(x_4{num}(2,:),2);
                elseif any(num == badTrials(p,:))
                    J_bad(p,num) = 1*(norm(x_5{num}(1,:),2)) + 1*(norm(x_5{num}(2,:),2)); ...
                        %                             + (norm(x_4{num}(1,:),2)) + norm(x_4{num}(2,:),2);
                end
            end
        end
        
    end
% save('HumanCost_J.mat','J_good','J_bad')
for i = 1:6
    temp = [];
    for m = 1:length(goodTrials(i,(goodTrials(i,:) ~= 0)))
        goodInd = goodTrials(i,m);
        for b = 1:length(badTrials(i,(badTrials(i,:) ~= 0)))
            badInd = badTrials(i,b);
%             sprintf('Pref Cost: %3.0f, Non Pref Cost: %3.0f',J_good(i,goodInd),J_bad(i,badInd));
            isMatching = any(J_good(i,goodInd) < J_bad(i,badInd));
            temp = [temp isMatching];
        end
    end
    validations{i} = sum(temp)/length(temp);
    %     sprintf('Subject %i: %i/%i',i,sum(temp),length(temp))
end
validations_all(costInd,:) = cell2mat(validations);
total(costInd) = sum(cell2mat(validations))/6;

end



%% run through all the patients and calculate the properties
%%%%% this script goes through the precompiled log structures and
%%%%% calculates the ZMP variables
close all;

% valLogsOnly = false;
% 
% if ~exist('all_logs')
%     if valLogsOnly
%         load([pwd,'/+ProcessLogs/LogData/all_val_logs.mat'])
%     else
%         load([pwd,'/+ProcessLogs/LogData/all_logs.mat'])
%     end
% end
% PatientNames = {'David','Marcus','Amy','Ke','Karena','Zoila'};
% 
% for p = 1:6
%     if exist('paths')
%         rmpath(paths.export_path);
%         rmpath(paths.sim_path);
%         rmpath(paths.load_path);
%         rmpath(paths.load_path);
%     end
%     [model,sys,paths] = ProcessLogs.loadModels(p);
%     
%     if valLogsOnly
%         logInds = 1:8;
%     else
%         logInds = 1:30;
%     end
%     
%     for num = logInds
%         if valLogsOnly
%             sprintf('Evaluating Subject %i, Validation %i',p,num)
%         else
%             sprintf('Evaluating Subject %i, Trial %i',p,num)
%         end
%         log = all_logs.(sprintf('%s_%i',PatientNames{p},num));
%         if ~isempty(log)
%             [numsteps,newStepInds] = ProcessLogs.getNumSteps(log);
%             ff_q = [log.odometry.extrapolated.pos;log.odometry.extrapolated.omega]; %x,y,z,yaw,pitch,roll
%             ff_dq = [log.odometry.extrapolated.vel;log.odometry.extrapolated.euler];
%             ff_q(3,:) = log.odometry.altitude;
%             
%             [ff_q,ff_dq,stanceFoot] = ProcessLogs.fixStanceFoot(ff_q,ff_dq,log,newStepInds,sys);
%             
%             ff_q = lowpass(ff_q',0.2);
%             ff_q = ff_q';
%             
%             ff_dq = lowpass(ff_dq',0.2);
%             ff_dq = ff_dq';
%             
%             stance = log.customFields(3,:) + 1; %stance = 1 when left stance,2 when right stance
%             swingFoot = [];
%             for i = 1:length(newStepInds)-1
%                 stepInds = newStepInds(i):newStepInds(i+1)-1;
%                 if stance(stepInds(1)) == 1
%                     for t = 1:length(stepInds)
%                         swingFoot(:,stepInds(t)) = h_RightSoleJoint_RightSS([zeros(6,1);log.joints.position(:,stepInds(t))], zeros(6,1));
%                         swingFoot(:,stepInds(t)) = swingFoot(:,stepInds(t)) + [ff_q(1,stepInds(t));ff_q(2:end,stepInds(t))];
%                     end
%                 else
%                     for t = 1:length(stepInds)
%                         swingFoot(:,stepInds(t)) = h_LeftSoleJoint_LeftSS([zeros(6,1);log.joints.position(:,stepInds(t))], zeros(6,1));
%                         swingFoot(:,stepInds(t)) = swingFoot(:,stepInds(t)) + [ff_q(1,stepInds(t));ff_q(2:end,stepInds(t))];
%                     end
%                 end
%             end
%             
%             q = [ff_q; log.joints.position];
%             
%             com = ProcessLogs.getCOM(model,paths,p,q);
%             com = lowpass(com',0.2);
%             com = com';
%             
%             com_dot = [zeros(3,1),diff(com,1,2)./diff(log.t)];
%             com_dot = lowpass(com_dot',0.2);
%             com_dot = com_dot';
%             com_ddot = [zeros(3,1),diff(com_dot,1,2)./diff(log.t)];
%             com_ddot = lowpass(com_ddot',0.2);
%             com_ddot = com_ddot';
%             
%             [cop,cop_metric_temp,dyn_metric_temp] = ProcessLogs.getCOP(log,ff_dq,com,stanceFoot,2);
%             cop_dot = [zeros(2,1),diff(cop,1,2)./diff(log.t)];
%             cop_dot = lowpass(cop_dot',0.2);
%             cop_dot = cop_dot';
%             
%             cop_ddot = [zeros(2,1),diff(cop_dot,1,2)./diff(log.t)];
%             cop_ddot = lowpass(cop_ddot',0.2);
%             cop_ddot = cop_ddot';
%             
%             p_dot = [zeros(2,1),diff(swingFoot([1,2],:),1,2)./diff(log.t)];
%             p_dot = lowpass(p_dot',0.2);
%             p_dot = p_dot';
%             
%             p_ddot = [zeros(2,1),diff(p_dot,1,2)./diff(log.t)];
%             p_ddot = lowpass(p_ddot',0.2);
%             p_ddot = p_ddot';
%             
%             %mistake somewhere in the frost model - offset sole for com calculation
%             ff_q_adj = ff_q + [0.1794;zeros(5,1)];
%             q_adj = [ff_q_adj; log.joints.position];
%             com = ProcessLogs.getCOM(model,paths,p,q_adj);
%             
%             patientLogData(p,num).log = log;
%             patientLogData(p,num).numsteps = numsteps;
%             patientLogData(p,num).newStepInds = newStepInds;
%             patientLogData(p,num).ff_q = ff_q;
%             patientLogData(p,num).ff_dq = ff_dq;
%             patientLogData(p,num).stanceFoot = stanceFoot;
%             patientLogData(p,num).swingFoot = swingFoot;
%             patientLogData(p,num).com = com;
%             patientLogData(p,num).com_dot = com_dot;
%             patientLogData(p,num).com_ddot = com_ddot;
%             patientLogData(p,num).cop = cop;
%             patientLogData(p,num).cop_dot = cop_dot;
%             patientLogData(p,num).cop_ddot = cop_ddot;
%             patientLogData(p,num).p_dot = p_dot;
%             patientLogData(p,num).p_ddot = p_ddot;
%         else
%             patientLogData(p,num).log = [];
%             patientLogData(p,num).numsteps = [];
%             patientLogData(p,num).newStepInds = [];
%             patientLogData(p,num).ff_q = [];
%             patientLogData(p,num).stanceFoot = [];
%             patientLogData(p,num).swingFoot = [];
%             patientLogData(p,num).com = [];
%             patientLogData(p,num).com_dot =[];
%             patientLogData(p,num).com_ddot = [];
%             patientLogData(p,num).cop = [];
%             patientLogData(p,num).cop_dot = [];
%             patientLogData(p,num).cop_ddot = [];
%             patientLogData(p,num).p_dot = [];
%             patientLogData(p,num).p_ddot = [];
%         end
%     end
%     rmpath(paths.export_path);
%     rmpath(paths.sim_path);
%     rmpath(paths.load_path);
%     rmpath(paths.anim_path);
% end
% 
% if valLogsOnly
%     save('patientValLogData.mat','patientLogData')
% else
%     save('patientLogData.mat','patientLogData')
% end
