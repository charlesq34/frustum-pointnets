function volume = cuboidVolume(bb)

dis = (bb([1 2 5 6],:)-bb([3 4 3 4],:)).^2;

volume = (bb(10,:)-bb(9,:)).*sqrt((dis(1,:)+dis(2,:)).*(dis(3,:)+dis(4,:)));