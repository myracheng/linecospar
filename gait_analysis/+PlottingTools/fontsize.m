function fontsize(varargin) %#ok<*AGROW>
narginchk(1,inf);

if(nargin==1)
    figHandle = [];
    fs = varargin{1};
    assert(~isempty(fs)&&isnumeric(fs)&&isscalar(fs),"If only one argument is provided, it must be a scalar number.")
    legendSize = fs;
    titleSize = fs;
    tickSize = fs;
    labelSize = fs;
else
    p = inputParser;
    p.addParameter('legend',[],@(x)~isempty(x)&&isnumeric(x)&&isscalar(x));
    p.addParameter('title',[],@(x)~isempty(x)&&isnumeric(x)&&isscalar(x));
    p.addParameter('tick',[],@(x)~isempty(x)&&isnumeric(x)&&isscalar(x));
    p.addParameter('label',[],@(x)~isempty(x)&&isnumeric(x)&&isscalar(x));
    p.addParameter('all',[],@(x)~isempty(x)&&isnumeric(x)&&isscalar(x));
    p.addParameter('handles',[],@(x)~isempty(x)&&isgraphics(x));
    p.parse(varargin{:});
    
    if(~isempty(p.Results.all))
        legendSize = p.Results.all;
        titleSize = p.Results.all;
        tickSize = p.Results.all;
        labelSize = p.Results.all;
    else
        legendSize = p.Results.legend;
        titleSize = p.Results.title;
        tickSize = p.Results.tick;
        labelSize = p.Results.label;
    end
    
    figHandle = p.Results.handles;
end
if(isempty(figHandle))
    allAxes = [];
    figHandle = findall(groot, 'Type', 'figure');
    if(isempty(figHandle))
        warning("No valid figure found.");
        return;
    end
    
    for i=1:length(figHandle)
        fh = figHandle(i);
        if(isvalid(fh))
            allAxes = [allAxes;findall(fh,'type','axes')];
        end
    end
    
    if(isempty(allAxes))
        warning("No valid axes found.");
        return;
    end
else
    if(isgraphics(figHandle,'figure'))
        if(length(figHandle)==1)
            assert(isvalid(figHandle),"Input is not a valid figure handle.")
            allAxes = findall(figHandle,'type','axes');
        else
            allAxes = [];
            figHandle = figHandle(:);
            for i=1:length(figHandle)
                fh = figHandle(i);
                if(~isvalid(fh))
                    warning("Figure handle #%i is not valid and will be ignored.",i);
                    continue;
                end
                allAxes = [allAxes;findall(fh,'type','axes')];
            end
        end
    elseif(isgraphics(figHandle,'axes'))
        allAxes = figHandle(:);
    else
        error("Input must be a valid figure/axes handle (or array of figure/axes handles).")
    end
end

for i=1:length(allAxes)
    h = allAxes(i);
    if(isvalid(h))
        if(~isempty(h.Legend) && ~isempty(legendSize))
            h.Legend.FontSize = legendSize;
        end
        if(~isempty(titleSize))
            h.Title.FontSize = titleSize;
        end
        if(~isempty(labelSize))
            h.XLabel.FontSize = labelSize;
            h.YLabel.FontSize = labelSize;
        end
        if(~isempty(tickSize))
            h.YAxis.FontSize = tickSize;
            h.XAxis.FontSize = tickSize;
        end
    else
        warning("Invalid axes handle, will be ignored.")
    end
end

end
