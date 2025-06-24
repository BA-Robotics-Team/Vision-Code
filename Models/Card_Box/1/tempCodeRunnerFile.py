from ultralytics import YOLO
Y=YOLO("runs\\BEx\\weights\\best.pt")
result=Y.predict("Models\\Card_Box\\1\\Dataset\\Valid\\images\\frame_00598.jpg" , save=True,show=True,verbose=False)
print(result[0].obb.xyxyxyxy)
print("Normalised: ",result[0].obb.xyxyxyxyAn)


def Get_Angle(ab):
    x1,y1,x2,y2,x3,y3,x4,y4=ab
    L1=((x22**2-x2**2)/2)
    L2=0



    if L1<L2:
        pass
    else:
        pass