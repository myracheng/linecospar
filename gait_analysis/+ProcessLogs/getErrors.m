function [x_1,x_2,x_3,x_4,x_5] = getErrors(newStepInds,stanceFoot,swingFoot,com,com_dot,com_ddot,cop,cop_dot,cop_ddot,p_dot,p_ddot,log,ff_q)
stance = log.stance + 1; %stance = 1 when left stance,2 when right stance

for i = 1:length(newStepInds)-3
    stepInds = newStepInds(i):newStepInds(i+1);
    COM_goal(:,stepInds) = repmat((stanceFoot([1,2],stepInds(end)+1)+stanceFoot([1,2],stepInds(1)))/2,1,length(stepInds));
    COM_goal(1,:) = COM_goal(1,:) + 0.1794;
    COM_step(:,stepInds) = com([1,2],stepInds);
    COM_vel(:,stepInds) = com_dot([1,2],stepInds);
    COM_accel(:,stepInds) = com_ddot([1,2],stepInds);
    COP_goal(:,stepInds) = COM_goal(:,stepInds);
    COP_step(:,stepInds) = cop([1,2],stepInds);
    COP_vel(:,stepInds) = cop_dot([1,2],stepInds);
       
    FF_gyro(:,stepInds) = ff_q([4,5],stepInds);
    COP_accel(:,stepInds) = cop_ddot([1,2],stepInds);
    Pos_ref(:,stepInds) = repmat(stanceFoot([1,2],stepInds(end)),1,length(stepInds));
    Pos(:,stepInds) = swingFoot([1,2],stepInds);
    
    Pos_vel(:,stepInds) = p_dot([1,2],stepInds);
    Pos_accel(:,stepInds) = p_ddot([1,2],stepInds);
    
%     figure
%     plot(COM_step(1,:),COM_step(2,:),'.')
%     [x, y, w, h] = soleToRect([stanceFoot(:,[stepInds(1),stepInds(end)+1]),swingFoot(:,stepInds(1))],3,log);
%     hold on
%     for fi = 1:3
%         foot = rectangle('Position',[x(fi),y(fi),w(fi),h(fi)]);
%     end
%     plot(COM_goal(1,end),COM_goal(2,end),'.','MarkerSize',10)
%     plot(Pos_ref(1,end),Pos_ref(2,end),'.','MarkerSize',10)
%     plot(Pos(1,:),Pos(2,:),'.')
%     plot(COP_step(1,:),COP_step(2,:))
    
end    
x_1 = (COM_goal - COM_step)/log.t(end); 
x_2 = COM_vel/log.t(end); 
x_3 = COP_vel/log.t(end); 
x_4 = (Pos_ref - Pos)/log.t(end);
x_5 = (COP_step - COM_step)/log.t(end);%FF_gyro/log.t(end);

end