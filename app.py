# Install flask 
from flask import Flask, render_template, request

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

app  = Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict(): 
    imagefile = request.files["imagefile"]
    image_path = './images/' + imagefile.filename
    imagefile.save(image_path)

    model = tf.keras.models.load_model('my_keras_model.h5', custom_objects={
                                   "KerasLayer": hub.KerasLayer})

    img_height = 256
    img_width = 256                               

    labels =  ['elefante', 'farfalla', 'mucca', 'pecora', 'scoiattolo']
    
    image = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    output = labels[np.argmax(predictions)]
    return render_template('index.html', prediction=output)


if __name__ == '__main__':
    app.run(port=3000, debug=True)