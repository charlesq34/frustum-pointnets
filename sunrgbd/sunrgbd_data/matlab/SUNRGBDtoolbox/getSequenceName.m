function sequenceName = getSequenceName(thispath,dataRoot)
    if ~exist('dataRoot','var'),
        dataRoot = '/n/fs/sun3d/data/';
    end
    sequenceName  = thispath(length(dataRoot):end);
    while sequenceName(1)=='/',sequenceName =sequenceName(2:end);end
    while sequenceName(end)=='/',sequenceName =sequenceName(1:end-1);end 
end