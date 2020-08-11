function ph = plotZMP_single(log,ax)

% colors to use in plotting
colorblind = [55,126,184; ...
    255, 127, 0;...
    77, 175, 74;...
    247, 129, 191;...
    166, 86, 40;...
    152, 78, 163;...
    153, 153, 153;...
    228, 26, 28;...
    222, 222, 0]/255;

% extract information from log
newStepInds = log.newStepInds;
stanceFoot = log.stanceFoot;
swingFoot = log.swingFoot;
com = log.com;
cop = log.cop;

% first step
stepInds = newStepInds(2):newStepInds(3);
COM_goal = repmat((stanceFoot([1,2],stepInds(end))+stanceFoot([1,2],stepInds(1)))/2,1,length(stepInds));
COM_goal(1,:) = COM_goal(1,:) + 0.1794;
COM_step = com([1,2],stepInds);
COP_step = cop([1,2],stepInds);
Pos_ref= repmat(stanceFoot([1,2],stepInds(end)),1,length(stepInds(1:end-1)));
Pos = swingFoot([1,2],stepInds(1:end-1));

% plot CoM traj of first step
plot(ax,COM_step(1,:),COM_step(2,:),'--','color',[colorblind(1,:),0.5])

% extract stance foot and swing foot stance positions
newStanceSoles = [stanceFoot(:,[stepInds(1),stepInds(end)+1]),swingFoot(:,stepInds(1))];

% get foot rectangle from stance foot position position
[x, y, w, h] = soleToRect(newStanceSoles);
hold on
for fi = 1:3
    foot = rectangle(ax,'Position',[x(fi),y(fi),w(fi),h(fi)],'EdgeColor',[colorblind(7,:),0.5]);
end

% plot ZMP variables for the first step
plot(ax,COM_goal(1,:),COM_goal(2,:),'p','MarkerSize',10,'color',[colorblind(1,:),0.5])
plot(ax,Pos_ref(1,:),Pos_ref(2,:),'p','MarkerSize',10,'color',[colorblind(3,:),0.5])
plot(ax,Pos(1,:),Pos(2,:),'--','color',[colorblind(3,:),0.5])
plot(ax,COP_step(1,:),COP_step(2,:),'--','color',[colorblind(2,:),0.5])

%repeat for second step
stepInds = newStepInds(3):newStepInds(4);
COM_goal = repmat((stanceFoot([1,2],stepInds(end))+stanceFoot([1,2],stepInds(1)))/2,1,length(stepInds));
COM_goal(1,:) = COM_goal(1,:) + 0.1794;
COM_step = com([1,2],stepInds);
COP_step = cop([1,2],stepInds);
Pos_ref = repmat(stanceFoot([1,2],stepInds(end)),1,length(stepInds(1:end-1)));
Pos = swingFoot([1,2],stepInds(1:end-1));

% plot ZMP variables with plot handles for making the legend later
ph(1) = plot(ax,COM_step(1,:),COM_step(2,:),'color',colorblind(1,:));
[x, y, w, h] = soleToRect([stanceFoot(:,[stepInds(1),stepInds(end)+1]),swingFoot(:,stepInds(1))]);
hold on
for fi = 1:3
    foot = rectangle(ax,'Position',[x(fi),y(fi),w(fi),h(fi)],'EdgeColor',colorblind(7,:));
end
ph(2) = plot(ax,COM_goal(1,:),COM_goal(2,:),'p','MarkerSize',10,'color',colorblind(1,:));
ph(3) = plot(ax,Pos_ref(1,:),Pos_ref(2,:),'p','MarkerSize',10,'color',colorblind(3,:));
ph(4) = plot(ax,Pos(1,:),Pos(2,:),'.','color',colorblind(3,:));
ph(5) = plot(ax,COP_step(1,:),COP_step(2,:),'color',colorblind(2,:));

end

%% helper function
function [x, y, w, h] = soleToRect(sole)

% load in foot dimension parameters
load([pwd,'/+PlottingTools/params']);

% extract x and y pos of sole
solexy = [sole(1,:);sole(2,:)];

% calculate top left, top right, bottom left, and bottom right corner of
% exoskeleton foot
tl = solexy+[params.rFoot.lengthToToe;params.rFoot.width];
tr = solexy+[params.rFoot.lengthToToe;-params.rFoot.width];
bl = solexy+[-params.rFoot.lengthToHeel;params.rFoot.width];
br = solexy+[-params.rFoot.lengthToHeel;-params.rFoot.width];

% output the bottom right position of the foot with the width and height of
% the exoskeleton foot (for the purposes of using rectangle())
x = br(1,:);
y = br(2,:);
w = sum([params.rFoot.lengthToToe,params.rFoot.lengthToToe])*ones(1,size(x,2));
h = 2*params.rFoot.width*ones(1,size(x,2));
end