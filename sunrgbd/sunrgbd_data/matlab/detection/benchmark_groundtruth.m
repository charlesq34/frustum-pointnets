function [groundtruthall,all_sequenceName] = benchmark_groundtruth(cls,path2gt,path2testAll)
        % get the ground truth box for this class
        try 
            a = load(fullfile(path2gt,'groundtruth.mat'));
        catch
            path2gt = '/n/fs/modelnet/SUN3DV2/prepareGT/Metadata/';
            a = load(fullfile(path2gt,'groundtruth.mat'));
        end
            
        if ~isempty(cls)
            pick = ismember({a.groundtruth.classname},cls);
            groundtruthall = a.groundtruth(pick);
        else
            groundtruthall = a.groundtruth;
        end
            
        if exist('path2testAll','var')&&~isempty(path2testAll)
            all_sequenceName = cell(1,length(path2testAll));
            for i =1:length(all_sequenceName)
                all_sequenceName{i} = getSequenceName(path2testAll{i},'/data/rqi/SUNRGBD/');
            end
            % hash ground truth image id and get the valid GT
            [validGT,GTimageid] = ismember({groundtruthall.sequenceName},all_sequenceName);
            groundtruthall = groundtruthall(validGT);
            GTimageid = GTimageid(validGT);
            for i =1:length(groundtruthall)
                groundtruthall(i).imageNum = GTimageid(i);
            end

        end
end
