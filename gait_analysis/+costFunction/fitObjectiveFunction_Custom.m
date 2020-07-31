close all;

PatientNames = {'1','2','3','4','5','6'};

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

w = [];
for leaveout = 1:6
    A = [];
    all = 1:6;
    alltried = all(all~=leaveout);
    for p = alltried
        goodInd = []; badInd = [];
        for num = logInds
            fprintf('Leaveout %i, Evaluating Subject %i, Validation %i . . . \n',leaveout,p,num)
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
                [x_1{num},x_2{num},x_3{num},x_4{num},x_5{num}] = ProcessLogs.getErrors(newStepInds,stanceFoot,swingFoot,com,com_dot,com_ddot,cop,cop_dot,cop_ddot,p_dot,p_ddot,log,ff_q);
                
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
                delta_5 = (norm(x_5{goodInd(m)}(1,:),2) + norm(x_5{goodInd(m)}(2,:),2)) ...
                    - (norm(x_5{badInd(i)}(1,:),2) + norm(x_5{badInd(i)}(2,:),2));
                A = [A; delta_1 delta_2 delta_3 delta_4 delta_5];
            end
        end
    end
    
    fun = @(w) norm(w,2);
    w0 = [1,1,1,1,1];
    Aeq = [1,1,1,1,1];
    beq = 1;
    %     A = [A; -eye(5)];
    w(leaveout,:) = fmincon(fun,w0,A,zeros(size(A,1),1),Aeq,beq);
    for ip = 1:5
        indstart = 1+(ip-1)*8;
        bcheck = A(indstart:indstart+7,:)*w(leaveout,:)';
        valcorrect(ip) = sum(bcheck < 0)/8;
    end
    valcorrect_all(leaveout,:) = valcorrect;
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
                
                [x_1{num},x_2{num},x_3{num},x_4{num},x_5{num}] = ProcessLogs.getErrors(newStepInds,stanceFoot,swingFoot,com,com_dot,com_ddot,cop,cop_dot,cop_ddot,p_dot,p_ddot,log,ff_q);
                
                wtemp = w(leaveout,:);
                if any(num == goodTrials(leaveout,:))
                    J_good(leaveout,num) = wtemp(1)*(norm(x_1{num}(1,:),2)) + wtemp(1)*(norm(x_1{num}(2,:),2)) ...
                        + wtemp(2)*(norm(x_2{num}(1,:),2)) + wtemp(2)*(norm(x_2{num}(2,:),2)) ...
                        + wtemp(3)*(norm(x_3{num}(1,:),2)) + wtemp(3)*(norm(x_3{num}(2,:),2)) ...
                        + wtemp(4)*(norm(x_4{num}(1,:),2)) + wtemp(4)*(norm(x_4{num}(2,:),2)) ...
                        + wtemp(5)*(norm(x_5{num}(1,:),2)) + wtemp(5)*(norm(x_5{num}(2,:),2));
                elseif any(num == badTrials(leaveout,:))
                    J_bad(leaveout,num) = wtemp(1)*(norm(x_1{num}(1,:),2)) + wtemp(1)*(norm(x_1{num}(2,:),2)) ...
                        + wtemp(2)*(norm(x_2{num}(1,:),2)) + wtemp(2)*(norm(x_2{num}(2,:),2)) ...
                        + wtemp(3)*(norm(x_3{num}(1,:),2)) + wtemp(3)*(norm(x_3{num}(2,:),2)) ...
                        + wtemp(4)*(norm(x_4{num}(1,:),2)) + wtemp(4)*(norm(x_4{num}(2,:),2)) ...
                        + wtemp(5)*(norm(x_5{num}(1,:),2)) + wtemp(5)*(norm(x_5{num}(2,:),2));
                end
            end
        end
    end
end

for i = 1:6
    temp = [];
    for m = 1:length(goodTrials(i,(goodTrials(i,:) ~= 0)))
        goodInd = goodTrials(i,m);
        for b = 1:length(badTrials(i,(badTrials(i,:) ~= 0)))
            badInd = badTrials(i,b);
            %             sprintf('Pref Cost: %3.0f, Non Pref Cost: %3.0f',J_good(i,goodInd),J_bad(i,badInd))
            isMatching = any(J_good(i,goodInd) < J_bad(i,badInd));
            temp = [temp isMatching];
        end
    end
    validations{i} = sum(temp)/length(temp);
end
validations_all = cell2mat(validations);
total = sum(cell2mat(validations))/6;

