PatientNames = {'1','2','3','4','5','6'};

close all;

combined = false;

% load the precompiled structure of ZMP variables
if ~exist('all_logs')
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

w = zeros(6,4);

for leaveout = 1:6
    A = [];
    all = 1:6;
    all(leaveout) = [];
    for p = all
        goodInd = []; badInd = [];
        for num = logInds
            %             sprintf('Leaveout %i, Evaluating Subject %i, Validation %i',leaveout,p,num)
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
                
                [x_1{num},x_2{num},x_3{num},x_4{num},~] = ProcessLogs.getErrors(newStepInds,stanceFoot,swingFoot,com,com_dot,com_ddot,cop,cop_dot,cop_ddot,p_dot,p_ddot,log,ff_q);
                if any(num == goodTrials(p,:))
                    goodInd = [goodInd num];
                elseif any(num == badTrials(p,:))
                    badInd = [badInd num];
                end
            end
        end
        for m = 1:length(goodInd)
            for i = 1:length(badInd)
                delta_1 = (norm(x_1{goodInd(m)}(1,:),2) + norm(x_1{goodInd(m)}(2,:),2)) ...
                    - (norm(x_1{badInd(i)}(1,:),2) + norm(x_1{badInd(i)}(2,:),2));
                delta_2 = (norm(x_2{goodInd(m)}(1,:),2) + norm(x_2{goodInd(m)}(2,:),2)) ...
                    - (norm(x_2{badInd(i)}(1,:),2) + norm(x_2{badInd(i)}(2,:),2));
                delta_3 = (norm(x_3{goodInd(m)}(1,:),2) + norm(x_3{goodInd(m)}(2,:),2)) ...
                    - (norm(x_3{badInd(i)}(1,:),2) + norm(x_3{badInd(i)}(2,:),2));
                delta_4 = (norm(x_4{goodInd(m)}(1,:),2) + norm(x_4{goodInd(m)}(2,:),2)) ...
                    - (norm(x_4{badInd(i)}(1,:),2) + norm(x_4{badInd(i)}(2,:),2));
                A = [A; delta_1 delta_2 delta_3 delta_4];
            end
        end
    end
    
    fun = @(w) norm(w,2);
    w0 = [1,1,1,1];
    Aeq = [1,1,1,1];
    beq = 1;
    A = [A; -eye(4)];
    w(leaveout,:) = fmincon(fun,w0,A,zeros(size(A,1),1),Aeq,beq);
    %     w(leaveout,:) = [8.04096e-4, 1.1618e-1, 3.8328e-3, 8.7918e-1];
    
    % leave one out patient
    for p = leaveout
        for num = logInds
            %             sprintf('Evaluating Subject %i, Validation %i',p,num)
            log = patientLogData(p,num);
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
            
            if ~isempty(log.t)
                
                [x_1{num},x_2{num},x_3{num},x_4{num},~] = ProcessLogs.getErrors(newStepInds,stanceFoot,swingFoot,com,com_dot,com_ddot,cop,cop_dot,cop_ddot,p_dot,p_ddot,log,ff_q);
                
                wtemp = w(leaveout,:);
                if any(num == goodTrials(leaveout,:))
                    J_good(leaveout,num) = wtemp(1)*(norm(x_1{num}(1,:),2)) + wtemp(1)*(norm(x_1{num}(2,:),2)) ...
                        + wtemp(2)*(norm(x_2{num}(1,:),2)) + wtemp(2)*(norm(x_2{num}(2,:),2)) ...
                        + wtemp(3)*(norm(x_3{num}(1,:),2)) + wtemp(3)*(norm(x_3{num}(2,:),2)) ...
                        + wtemp(4)*(norm(x_4{num}(1,:),2)) + wtemp(4)*(norm(x_4{num}(2,:),2));
                elseif any(num == badTrials(leaveout,:))
                    J_bad(leaveout,num) = wtemp(1)*(norm(x_1{num}(1,:),2)) + wtemp(1)*(norm(x_1{num}(2,:),2)) ...
                        + wtemp(2)*(norm(x_2{num}(1,:),2)) + wtemp(2)*(norm(x_2{num}(2,:),2)) ...
                        + wtemp(3)*(norm(x_3{num}(1,:),2)) + wtemp(3)*(norm(x_3{num}(2,:),2)) ...
                        + wtemp(4)*(norm(x_4{num}(1,:),2)) + wtemp(4)*(norm(x_4{num}(2,:),2));
                end
            end
        end
    end
end


% save('StevensCost_J.mat','w','J_good','J_bad')

% check if checked J associate with good/bad trials for that subject
% load('StevensCost_J.mat')
for i = 1:6
    temp = [];
    for m = 1:length(goodTrials(i,(goodTrials(i,:) ~= 0)))
        goodInd = goodTrials(i,m);
        for b = 1:length(badTrials(i,(badTrials(i,:) ~= 0)))
            badInd = badTrials(i,b);
            sprintf('Pref Cost: %3.0f, Non Pref Cost: %3.0f',J_good(i,goodInd),J_bad(i,badInd))
            isMatching = any(J_good(i,goodInd) < J_bad(i,badInd));
            temp = [temp isMatching];
        end
    end
    validations{i} = sum(temp)/length(temp);
end
validations
val_avg = sum(cell2mat(validations))/6