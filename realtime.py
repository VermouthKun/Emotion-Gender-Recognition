import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading

emotion_model = load_model('Emotion1.h5')  

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

camera_active = False

def update_frame():
    global camera_active
    if camera_active:
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]

                roi_gray_resized = cv2.resize(roi_gray, (48, 48))
                roi_gray_resized = roi_gray_resized.astype('float32') / 255.0
                roi_gray_resized = np.expand_dims(roi_gray_resized, axis=0)
                roi_gray_resized = np.expand_dims(roi_gray_resized, axis=-1)

                emotion_prediction = emotion_model.predict(roi_gray_resized)
                emotion_index = np.argmax(emotion_prediction)
                emotion = emotion_labels[emotion_index]

                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)

            lbl_video.imgtk = imgtk
            lbl_video.configure(image=imgtk)

        lbl_video.after(10, update_frame)

def run_recognition():
    global cap, camera_active
    camera_active = True
    cap = cv2.VideoCapture(0)
    update_frame()

def stop_recognition():
    global camera_active
    camera_active = False

def quit_program():
    global cap, camera_active
    camera_active = False
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    root.quit()

root = tk.Tk()
root.title("Posts and Telecommunications Institute of Technology")
root.geometry("900x700") 
root.configure(background="green")

img_path = "C:/Users/anhku/Desktop/Emotion-Gender-Recognition/img/images.png"  
logo_icon = Image.open(img_path)
logo_icon = logo_icon.resize((32, 32), Image.LANCZOS) 
logo_icon_tk = ImageTk.PhotoImage(logo_icon)
root.iconphoto(False, logo_icon_tk)  

label_title = tk.Label(root, text="PREDICT EMOTION BASED ON HUMAN FACES", font=("Lato", 20, 'bold'), bg="green", fg="red")
label_title.pack(pady=10)

label_subtitle = tk.Label(root, text="Artificial Intelligence\nSon Anh Le and Yumeno Tanaka", font=("Lato", 14, 'bold'), bg="green", fg="white")
label_subtitle.pack()

lbl_video = tk.Label(root)
lbl_video.pack(pady=10)

button_frame = tk.Frame(root, bg="green")
button_frame.pack(pady=10)

start_button = tk.Button(button_frame, text="START", font=("Lato", 16, 'bold'), bg="blue", fg="white", command=lambda: threading.Thread(target=run_recognition).start())
start_button.grid(row=0, column=0, padx=10)

stop_button = tk.Button(button_frame, text="STOP", font=("Lato", 16, 'bold'), bg="blue", fg="white", command=stop_recognition)
stop_button.grid(row=0, column=1, padx=10)

exit_button = tk.Button(button_frame, text="EXIT", font=("Lato", 16, 'bold'), bg="blue", fg="white", command=quit_program)
exit_button.grid(row=0, column=2, padx=10)

root.mainloop()
