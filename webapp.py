"""
Simple app to upload an image via a web form 
and view the inference results on the image in the browser.
"""
import argparse
import io
from PIL import Image
import datetime

import torch
import cv2
import numpy as np
# import tensorflow as tf
from re import DEBUG, sub
from flask import Flask, jsonify, render_template, request, redirect, send_file, url_for, Response
from werkzeug.utils import secure_filename, send_from_directory
import os
import subprocess
from subprocess import Popen
import re
import requests
import shutil
import time
import base64
import csv

from flask_bootstrap import Bootstrap5

app = Flask(__name__)

# root theme
app.config['BOOTSTRAP_BOOTSWATCH_THEME'] = 'flatly'

Bootstrap5(app)

current_selected_file = None

# @app.template_filter('b64encode')
# def base64_encode(data):
#     return base64.b64encode(data).decode('utf-8')


@app.route("/")
def hello_world():
    image_names = os.listdir('static/assets/home_img')
    return render_template('index.html', image_names=image_names)

def read_csv_file(file_path):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        for row in reader:
            data.append(row)
    return headers, data

@app.route("/training")
def training_result():
    file_path = 'data/result.csv'
    headers, csv_data = read_csv_file(file_path)
    return render_template('training_result.html', headers=headers, csv_data=csv_data)

@app.route("/demo")
def instance_demo():
    return render_template('instance_demo.html')

@app.route("/recognize", methods=['POST'])
def recognize_file():
    uploaded_file = request.files['file']
    if uploaded_file:
        filename = uploaded_file.filename
        file_path = os.path.join('uploads', filename)
        uploaded_file.save(file_path)
        
        file_extension = filename.rsplit('.', 1)[1].lower()
            
        if file_extension in ['png', 'mp4']:
            recognize_file.filename = filename
            print("printing predict_img :::::: ", recognize_file)
            
            process = Popen(["python", "detect.py", '--source', file_path, "--weights","after.pt"], shell=True)
            # detect_before
            process_before = Popen(["python", "detect_before.py", '--source', file_path, "--weights","before.pt"], shell=True)

            if file_extension == 'mp4':
                process.communicate()
                process_before.communicate()
                
            process.wait()
            
            folder_path = 'runs/detect'
            latest_subfolder = get_latest_folder(folder_path)    
            image_path = f"{latest_subfolder}/{filename}" 
            print("printing image_path :::::: ", image_path)
            return jsonify({'recognized_image_path': image_path})
        else:
          return jsonify({'error': f"The file '{filename}' is not supported."})  
    else:
        return jsonify({'error': 'The file was not found.'})

@app.route("/compare")
def instance_compare():
    before_folder_path = 'runs/detect_before'
    before_image = get_latest_file(before_folder_path)
    after_folder_path = 'runs/detect'
    after_image = get_latest_file(after_folder_path)
    return render_template('instance_compare.html', before_image=before_image, after_image=after_image)

@app.route("/about")
def about_system():
    return render_template('about_system.html')

def get_latest_folder(folder_path):
    sub_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    if not sub_folders:
        return None
    
    latest_subfolder = max(sub_folders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))
    return os.path.join(folder_path, latest_subfolder).replace('\\', '/')

def get_latest_file(folder_path):
    latest_subfolder_path = get_latest_folder(folder_path)
    if not latest_subfolder_path:
        return None

    files = [os.path.join(latest_subfolder_path, f) for f in os.listdir(latest_subfolder_path) if os.path.isfile(os.path.join(latest_subfolder_path, f))]
    if not files:
        return None

    latest_file = max(files, key=os.path.getctime)
    return latest_file.replace('\\', '/')


# function for accessing rtsp stream
# @app.route("/rtsp_feed")
# def rtsp_feed():
    # cap = cv2.VideoCapture('rtsp://admin:hello123@192.168.29.126:554/cam/realmonitor?channel=1&subtype=0')
    # return render_template('index.html')


# Function to start webcam and detect objects

# @app.route("/webcam_feed")
# def webcam_feed():
    # #source = 0
    # cap = cv2.VideoCapture(0)
    # return render_template('index.html')

# function to get the frames from video (output video)

def get_frame():
    if not hasattr(recognize_file, 'filename'):
        return
    folder_path = 'runs/detect'
    latest_subfolder = get_latest_folder(folder_path) 
    filename = recognize_file.filename    
    image_path = f"{latest_subfolder}/{filename}"
    video = cv2.VideoCapture(image_path)  # detected video path
    #video = cv2.VideoCapture("video.mp4")
    while True:
        success, image = video.read()
        if not success:
            break
        ret, jpeg = cv2.imencode('.png', image)   
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')   
        time.sleep(0.1)  #control the frame rate to display one frame every 100 milliseconds: 


# function to display the detected objects video on html page
@app.route("/video_feed")
def video_feed():
    return Response(get_frame(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



#The display function is used to serve the image or video from the folder_path directory.
@app.route('/<path:filename>')
def display(filename):
    # folder_path = 'runs/detect'
    # sub_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]    
    # latest_subfolder = max(sub_folders, key=lambda x: os.path.getctime(os.path.join(folder_path, x)))    
    # directory = f"{folder_path}/{latest_subfolder}" 
    # filename = recognize_file.filename 
    file_extension = filename.rsplit('.', 1)[1].lower()
    #print("printing file extension from display function : ",file_extension)
    environ = request.environ
    if file_extension == 'png':      
        return send_from_directory('', filename, environ)
    else:
        return "Invalid file format"


if __name__ == "__main__":
    app.run(debug=True)
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()
    model = torch.hub.load('.', 'custom','best_246.pt', source='local')
    model.eval()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat

