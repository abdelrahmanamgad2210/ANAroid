import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
import pandas as pd
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import subprocess
import shutil
from flask_cors import CORS, cross_origin
import time
from threading import Thread



tf.keras.utils.get_custom_objects()['KerasLayer'] = hub.KerasLayer
model = tf.keras.models.load_model('models/CNN_GRU.h5')

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

global featureFile
featureFile = None

global extraction_status
extraction_status= {'completed': False}
max_words = 10000
max_len = 100

def tokenize_and_sequence_only(full_texts, texts, max_len, max_words):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(full_texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences, maxlen=max_len)
    word_index = tokenizer.word_index
    return tokenizer, word_index, data

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/')
def home():
    return render_template('predict.html')

@app.route('/api/upload_file',  methods=['POST'])
def upload_file():
    print("Da vao analysic_file")
    if 'file' not in request.files:
        return {"isCompleted": False, "conclusion": "File not found"}

    file = request.files['file']
    if file.filename == '':
        return {"isCompleted": False, "conclusion": "Filename error"}

    if file: 
        fileName = ''.join(file.filename.split(".")[:-1])
        global featureFile
        featureFile = fileName + "-analysis.json"
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
        return {"message": "Your file has been saved"}

@app.route('/api/run_extraction',  methods=['GET'])
def run_extraction():
    # Run extraction
    def run_command():
        result = subprocess.run(['bash', os.path.join(app.root_path, 'AndroPyTool-Autorun', 'extraction.sh'), os.path.join(app.root_path, 'static', 'files'), os.path.join(app.root_path, 'reports')])
        # Update the status once the extraction is complete
        extraction_status['completed'] = True
    
    t1 = Thread(target=run_command)
    t1.start()
    
    return {"message": "Successfully started extraction"}

@app.route('/api/check_status', methods=['GET'])
def check_status():
    global extraction_status
    return jsonify(extraction_status)

@app.route('/api/analysic_result', methods=['GET'])
def analysic_result():
    global extraction_status
    if not extraction_status['completed']:
        return jsonify({"message": "Extraction not completed yet"}), 400
    
    feature_file_path = os.path.join(app.root_path, 'reports', 'Features_files', featureFile)
    
    try:
        with open(feature_file_path, "r") as f:
            json_data = json.load(f)
    except FileNotFoundError:
        return jsonify({"message": "Feature file not found"}), 404
    except json.JSONDecodeError:
        return jsonify({"message": "Error decoding JSON"}), 400
    
    texts = ' '.join([str(value) for key, value in json_data.items()])

    print(f"Extracted texts: {texts}")

    file_path = "/home/abood/FINALPROJECT/df_train_cleaned.csv"
    df_train_loaded = pd.read_csv(file_path)
    full_texts = df_train_loaded['fileContent'].tolist()

    print(f"Full texts for training (sample): {full_texts[:1]}")

    tokenizer, word_index, vectors = tokenize_and_sequence_only(full_texts, [texts], max_len, max_words)

    print(f"Tokenized and padded vectors: {vectors}")

    predictions = model.predict(vectors)
    print(f"Model predictions: {predictions}")

    if predictions.size == 0:
        return jsonify({"message": "No predictions made"}), 500
    
    predicted_classes = np.argmax(predictions, axis=1)
    class_names = ['banking', 'riskware', 'benign', 'adware']
    predicted_class = class_names[predicted_classes[0]]

    print(f"Predicted class: {predicted_class}")

    response = {
        "isCompleted": True,
        "conclusion": predicted_class
    }
    
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
