function plotZMP_animated(log,ax)

% color palate to use in plots:
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
stance = log.stance + 1; %change of notation: 1 is left stance, 2 is right stance
stanceFoot = log.stanceFoot;
newStepInds = log.newStepInds;
com = log.com;
cop = log.cop;

% add double support stance to stance information
stanceNew = stance; %new variable to use to detect new stance impact events
doublesupport = find(log.feet.left.F(3,:) > 50 & log.feet.right.F(3,:) > 50);
stance(doublesupport) = 3;

% get all of the rectangle geometries for each stance foot position
[x, y, w, h] = soleToRect(stanceFoot);

% go through each step (up to 20 steps max)
for i = 1:min(length(newStepInds)-1,20)
    
    % get the indice corresponding to the current step
    indtoplot = newStepInds(i);
    
    % plot the stance foot rectangle in blue if left stance, red if right
    % stance
    if stanceNew(indtoplot) == 1 %left stance blue
        rectangle(ax,'Position',[x(indtoplot), y(indtoplot), w(indtoplot), h(indtoplot)],'FaceColor',[colorblind(7,:),0.3]); %[x y w h]
    elseif stanceNew(indtoplot) == 2 %right  stance red
        rectangle(ax,'Position',[x(indtoplot), y(indtoplot), w(indtoplot), h(indtoplot)],'FaceColor',[colorblind(7,:),0.3]); %[x y w h]
    end
    
    % plot an initial marker to be used as the marker for the legend later
    if stanceNew(indtoplot) == 1
        lp{1} = plot(0,0,'.','MarkerSize',10,'color',colorblind(1,:));
        rp{1} = plot(0,0,'.','MarkerSize',10,'color',colorblind(2,:));
        bp{1} = plot(0,0,'.','MarkerSize',10,'color',colorblind(3,:));
        copax{1} = plot(0,0,'.','MarkerSize',10,'color',colorblind(5,:));
    end
    
    % plot the CoM and CoP trajectories for the current step
    for t = 1:size(com(:,newStepInds(i):newStepInds(i+1)-1),2)
        indtoplot = newStepInds(i)-1+t;
        
        % CoP trajectory is always brown
        % CoM trajectory is colored differently depending on if left
        % stance, right stance, or double stance
        if stance(indtoplot) == 1
            plot(ax,com(1,indtoplot),com(2,indtoplot),'.','color',colorblind(1,:));
            plot(ax,cop(1,indtoplot),cop(2,indtoplot),'.','color',colorblind(5,:));
        elseif stance(indtoplot) == 2
            plot(ax,com(1,indtoplot),com(2,indtoplot),'.','color',colorblind(2,:));
            plot(ax,cop(1,indtoplot),cop(2,indtoplot),'.','color',colorblind(5,:));
        else
            plot(ax,com(1,indtoplot),com(2,indtoplot),'.','color',colorblind(3,:));
            plot(ax,cop(1,indtoplot),cop(2,indtoplot),'.','color',colorblind(5,:));
        end
        hold on
        
        % this is necessary for animation. Can comment out if you don't
        % want animated plotting
        drawnow limitrate
    end
end

% add legend
legend([lp{1},rp{1},bp{1},copax{1}],{'CoM Left Stance','CoM Right Stance','CoM Double Support','CoP'},'Orientation','horizontal','Location','South','NumColumns',2)
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