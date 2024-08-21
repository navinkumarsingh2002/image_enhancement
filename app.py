from flask import Flask, request, render_template, send_file
import os
from model import load_model
from utils import enhance_image

app = Flask(__name__)
model = load_model('enhancement_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)
    
    output_path = os.path.join('enhanced', file.filename)
    enhance_image(model, file_path, output_path)
    
    return send_file(output_path)

if __name__ == '__main__':
    app.run(debug=True)
