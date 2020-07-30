function plotZMP(stanceFoot,newStepInds,com,cop,log,dim,ax)
%dim = 1: plot ZMP in x dimension
%dim = 2: plot ZMP in y dimension
%dim = 3: plot ZMP in both dimensions

colorblind = [55,126,184; ...
              255, 127, 0;...
              77, 175, 74;...
              247, 129, 191;...
              166, 86, 40;...
              152, 78, 163;...
              153, 153, 153;...
              228, 26, 28;...
              222, 222, 0]/255;
          
% h = animatedline;
stance = log.stance + 1; 
% tempstance = stance;
% stance(tempstance == 1) = 2;
% stance(tempstance == 2) = 1; %stance = 2 when left stance,1 when right stance

stanceNew = stance;

doublesupport = find(log.feet.left.F(3,:) > 50 & log.feet.right.F(3,:) > 50);
stance(doublesupport) = 3;

[x, y, w, h] = soleToRect(stanceFoot,dim,log);

for i = 1:min(length(newStepInds)-1,20)
    indtoplot = newStepInds(i);
    if stanceNew(indtoplot) == 1 %left stance blue
        rectangle(ax,'Position',[x(indtoplot), y(indtoplot), w(indtoplot), h(indtoplot)],'FaceColor',[colorblind(7,:),0.3]); %[x y w h]
    elseif stanceNew(indtoplot) == 2 %right  stance red
        rectangle(ax,'Position',[x(indtoplot), y(indtoplot), w(indtoplot), h(indtoplot)],'FaceColor',[colorblind(7,:),0.3]); %[x y w h]
    end
        hold on
    if stanceNew(indtoplot) == 1
       lp{1} = plot(0,0,'.','MarkerSize',10,'color',colorblind(1,:));
       rp{1} = plot(0,0,'.','MarkerSize',10,'color',colorblind(2,:));
       bp{1} = plot(0,0,'.','MarkerSize',10,'color',colorblind(3,:));
       copax{1} = plot(0,0,'.','MarkerSize',10,'color',colorblind(5,:));
    end
    a = tic;
    for t = 1:size(com(:,newStepInds(i):newStepInds(i+1)-1),2)
        indtoplot = newStepInds(i)-1+t;
        if dim == 3
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
            drawnow limitrate
%             b = toc(a);
%             if b > (0.0001)
%                 drawnow 
%                 a = tic;
%             end
        elseif dim == 2
            plot(ax,log.t,com(2,indtoplot),'.')
            hold on
        end
    end
end

legend([lp{1},rp{1},bp{1},copax{1}],{'CoM Left Stance','CoM Right Stance','CoM Double Support','CoP'},'Orientation','horizontal','Location','South','NumColumns',2)
end

%color convex region:


function [x, y, w, h] = soleToRect(sole,dim,log)
load([pwd,'/+PlottingTools/params.mat'])

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

function P = getConvexRegion(stance1,stance2)

tl1 = stance1+[params.rFoot.lengthToToe;params.rFoot.width];
tr1 = stance1+[params.rFoot.lengthToToe;-params.rFoot.width];
bl1 = stance1+[-params.rFoot.lengthToHeel;params.rFoot.width];
br1 = stance1+[-params.rFoot.lengthToHeel;-params.rFoot.width];

tl2 = stance2+[params.rFoot.lengthToToe;params.rFoot.width];
tr2 = stance2+[params.rFoot.lengthToToe;-params.rFoot.width];
bl2 = stance2+[-params.rFoot.lengthToHeel;params.rFoot.width];
br2 = stance2+[-params.rFoot.lengthToHeel;-params.rFoot.width];

if tl1 >= tl2
    P = [tl1';tl2';tr2';br2';br1';bl1];
else
    P = [tl1';tr1';tr2';br2';bl2';bl1'];
end


end