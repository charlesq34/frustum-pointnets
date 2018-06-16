function sustr = findsubstring(str,strartstr,endstr)
         [ind1,ind2]=regexp(str, strartstr);
         str = str(ind2+1:end);
         [ind1,ind2]=regexp(str, endstr);
         sustr = str(1:ind1-1);
end