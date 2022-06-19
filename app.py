import cv2
import numpy as np 
import time
from tkinter import *
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from keras.models import load_model
from PIL import Image,ImageTk

shape_model = load_model('shapeface_model.h5')
age_model = load_model('age_model.h5')
gender_model = load_model('gender_model.h5')

shape_labels=['Heart','Oblong','Oval','Round','Square']
age_labels = ['18','19','20','21','22','23','24','25','26','27']
gender_labels =['Female','Male']

age=""
gender=""
shape=""

#Khoi tao giao dien gui 
tk=Tk() 
tk.title("Face Regconition") 
tk.geometry("800x500+0+0") 
tk.resizable(0,0) 
tk.configure(background="white") 
content1="Age, Gender, and Facial Shape Regconition"
lb01=Label(tk,fg="green",bg="white",font="Times 18",text=content1) 
lb01.pack()
lb01.place(x=210,y=10) 

#Hien thi ten khung hinh

lb04=Label(tk,text="Webcam",font="Times 12",fg="blue",bg="white")
lb04.pack()
lb04.place(x=380,y=90)

##### Hien thi so luong loai### 
lb11=Label(tk,fg="green",bg="white",font="Times 18",text="Age: ") 
lb11.pack()
lb11.place(x=90,y=450) 

lb12=Label(tk,fg="green",bg="white",font="Times 18",text="Gender: ") 
lb12.pack()
lb12.place(x=280,y=450) 

lb13=Label(tk,fg="green",bg="white",font="Times 18",text="Shape: ") 
lb13.pack()
lb13.place(x=520,y=450) 


#Khoi tao camera 
capture = cv2.VideoCapture(0) 

def close_window():
    tk.destroy()

def ConvertImage(convert_img):
    image = convert_img[:,80:(80+480)]
    image = cv2.resize(image, dsize =(128,128))
    image = np.expand_dims(image, axis=0)
    return image

def Regconition(reg_img):
    #Shape Face
    shape_predict = shape_model.predict(reg_img)
    shape_label= shape_labels[np.argmax(shape_predict)]


    #Age
    age_predict = age_model.predict(reg_img)
    age_label= age_labels[np.argmax(age_predict)]

    #Gender
    gender_predict = gender_model.predict(reg_img)
    gender_label = gender_labels[np.argmax(gender_predict)]

    return age_label, gender_label, shape_label



while capture.isOpened():
    ret, image_ori = capture.read() 
    cv2.imwrite('image_ori.jpg',image_ori) 
    imagelg=Image.open('image_ori.jpg') 
    imagelg=imagelg.resize((400,300),Image.ANTIALIAS) 
    imagelg=ImageTk.PhotoImage(imagelg) 
    lb05=Label(image=imagelg)
    lb05.image=imagelg 
    lb05.pack()  
    lb05.place(x=200,y=110)
    tk.update() 
    image = ConvertImage(image_ori)
    age, gender, shape = Regconition(image)   
    lb21=Label(tk,fg="green",bg="white",font="Times 18",text=age) 
    lb21.pack() 
    lb21.place(x=140,y=450) 
    lb22=Label(tk,fg="green",bg="white",font="Times 18",text=gender) 
    lb22.pack()
    lb22.place(x=360,y=450) 
    lb23=Label(tk,fg="green",bg="white",font="Times 18",text=shape) 
    lb23.pack()
    lb23.place(x=590,y=450) 
    age = ""
    gender=""
    shape= ""


    
    if cv2.waitKey(1) == ord('q'):
        close_window() 
cv2.destroyAllWindows() 
tk.mainloop()
