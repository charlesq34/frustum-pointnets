function fileStr = file2string(fname)
    fileStr = '';
    fid = fopen(fname,'r');
    tline = fgetl(fid);
    while ischar(tline)
        fileStr = [fileStr sprintf('\n') tline];
        tline = fgetl(fid);
    end
    fclose(fid);
end