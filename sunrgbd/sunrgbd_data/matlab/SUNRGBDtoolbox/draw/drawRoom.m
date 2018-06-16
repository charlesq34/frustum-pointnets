function drawRoom(roomLayout,color,lineWidth,maxhight)
totalPoints  = size(roomLayout,2);
bottompoints= roomLayout(:,1:totalPoints/2);
toppoints = roomLayout(:,totalPoints/2+1:totalPoints);
toppoints(3,toppoints(3,:)>maxhight)=maxhight;
bottompoints(3,bottompoints(3,:)>maxhight)=maxhight;
[~,ind] = min(toppoints(2,:));
for i =1:length(toppoints)-1
    if i~=ind&&i+1~=ind
    vis_line(toppoints(:,i)', toppoints(:,i+1)', color, lineWidth);
    vis_line(bottompoints(:,i)', bottompoints(:,i+1)', color, lineWidth);
    vis_line(toppoints(:,i)', bottompoints(:,i)', color, lineWidth);
    end
end
    if 1~=ind&&size(toppoints,2)~=ind
        vis_line(toppoints(:,end)', toppoints(:,1)', color, lineWidth);
        vis_line(bottompoints(:,end)', bottompoints(:,1)', color, lineWidth);
        vis_line(toppoints(:,end)', bottompoints(:,end)', color, lineWidth);
    end
end