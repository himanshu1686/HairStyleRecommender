from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image

# Keras
# from keras.applications.imagenet_utils import preprocess_input, decode_predictions
# from keras.models import load_model
# from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# MODEL_PATH = 'models/model_resnet.h5'
# MODEL_PATH = 'models/retrained_graph.pb'


# Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
# print('Model loaded. Check http://127.0.0.1:5000/')

def load_image(filename):
    #Read in the image_data to be classified."""
    # im = Image.open(filename)
    # new_img = im.resize( (299,299) ) 
    # new_img.show()
    # new_img.save( './uploads/current_pred.jpg' )
    # print( new_img.info )
    # return tf.gfile.FastGFile('./uploads/current_pred.jpg', 'rb').read()
    return tf.gfile.FastGFile(filename, 'rb').read()

def load_labels(filename):
    #Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
    #Unpersists graph from file as default graph."""
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


labels_path='./models/retrained_labels.txt'
graph_path='./models/retrained_graph.pb'
input_layer='DecodeJpeg/contents:0'
output_layer='final_result:0'
num_top_predictions=1

def run_graph(img_path, labels, input_layer_name, output_layer_name, num_top_predictions):
    with tf.Session() as sess:
        print("[Info : In Model Predect . ]")
        image_data=load_image(img_path)
        softmax_tensor=sess.graph.get_tensor_by_name(output_layer_name)
        predictions,=sess.run(softmax_tensor, {input_layer_name: image_data})
        print(predictions)
        # Sort to show labels in order of confidence             
    sess.close()
    return  predictions       

def model_predict( img_path ):
    # img = image.load_img(img_path, target_size=(224, 224))
    print("[Info : In Model Predect . ]")
    labels = load_labels(labels_path)
    print("labels : " , labels )
    load_graph(graph_path)
    predictions = run_graph(img_path,labels,input_layer,output_layer,num_top_predictions)
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    print(top_k)
    finalPred =''
    for node_id in top_k:
        predicted_label = labels[node_id]
        final_pred = predicted_label
        score = predictions[node_id]
        #outfile.write(test[i]+','+human_string+'\n')
    return final_pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print("[Info :file uploaded ]")
        # Make prediction
        pred = model_predict(file_path)
        
        print(pred)
        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        # result = str(pred_class[0][0][1])               # Convert to string
        return pred
    return None


if __name__ == '__main__':
    app.run(debug=True)

