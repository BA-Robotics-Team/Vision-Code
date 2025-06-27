import math as m
import socket 

def Ikinematics(x,y,z):

  link1=12
  link2=12
  
  #Motor 0
  theta0=(m.atan2(y,x))*180/(m.pi) #Motor 0 Angle

  #Motor 1
  W = m.sqrt((x**2) + (y**2))
  A = m.sqrt((z**2) + (W**2))

  gamma = (m.acos((link1**2 + A**2 - link2**2)/(2*link1*A)))*180/(m.pi)
  tou = (m.atan2(z,W))*180/(m.pi)
  theta1 = gamma + tou #Motor 1  Angle

  #Motor 2
  alpha=(m.acos((link1**2 + link2**2 - A**2)/(2*link1*link2)))*180/(m.pi)
    
  theta2=180-alpha #Motor 2 Angle  
  return [theta0, theta1, theta2]

def TCP_Comm(s,parameters):
  s.send(f"{parameters[0]:.2f},{parameters[1]:.2f},{parameters[2]:.2f}".encode())

if __name__ =="__main__": 
  s=socket.socket()
  s.connect(('192.168.31.100', 12345)) 
 # Replace with your Pi's IP
  while True:
   x=float(input("Enter x coordinate:"))
   y=float(input("Enter y coordinate:"))
   z=float(input("Enter z coordinate:"))
  
   if x == 100:break
   if y == 100:break
   if z == 100:break
  
   actuations=Ikinematics(x,y,z)
   print(actuations)
   TCP_Comm(s,actuations)