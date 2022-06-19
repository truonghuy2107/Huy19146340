from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

shape_model = load_model("shapeface_model.h5")
age_model = load_model("age_model.h5")
gender_model = load_model("gender_model.h5")

shape_labels=['Heart','Oblong','Oval','Round','Square']
age_labels = ['18','19','20','21','22','23','24','25','26','27']
gender_labels =['Female','Male']

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    img = frame
    img = img[:,80:(80+480)]


    frame = cv2.resize(img, dsize =(128,128))
    tensor = np.expand_dims(frame, axis=0)

    #Shape Face
    shape_predict = shape_model.predict(tensor)
    shape_label=shape_labels[np.argmax(shape_predict)] 
    cv2.putText(img, shape_label, (190, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,0), 2)

    #Age
    age_predict = age_model.predict(tensor)
    age_label=age_labels[np.argmax(age_predict)] 
    cv2.putText(img, age_label,(10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255), 2)
    #Gender
    gender_predict = gender_model.predict(tensor)
    gender_label=gender_labels[np.argmax(gender_predict)] 
    cv2.putText(img, gender_label, (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0,0), 2)

    cv2.imshow('Face Detector', img)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()