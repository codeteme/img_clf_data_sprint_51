#import lib
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from matplotlib.pyplot import imread
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import ImageTk, Image

#import model
print('_________________________________')
print('..........Start loading..........')

model = tf.keras.models.load_model('../my_keras_model.h5', custom_objects={
                                   "KerasLayer": hub.KerasLayer})

print('_________________________________')
print('...........Model Loaded..........')
print('_________________________________')

# List to label all class.
labels = ['Cat', 'Dog']

# initialise GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Animal Classification')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

batch_size = 32
img_height = 256
img_width = 256

#To classify an image
def classify(file_path):
    global label_packed
    image = imread(file_path)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[img_height, img_width])
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    sign = labels[np.argmax(pred)]
    print(sign)
    label.configure(foreground='#011638', text=sign)

#Button
def show_classify_button(file_path):
    classify_b = Button(top, text="Classify Image",
                        command=lambda: classify(file_path),
                        padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white',
                         font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

#Upload image
def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
                            (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload = Button(top, text="Upload an image",
                command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white',
                 font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Image Classification",
                pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()