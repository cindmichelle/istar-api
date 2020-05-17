import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

UPLOAD_FOLDER_INPUT = './uploads/inputs'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER_INPUT'] = UPLOAD_FOLDER_INPUT

# model = model.load


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return 'Index Page'


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(
                app.config['UPLOAD_FOLDER_INPUT'], filename))
            # call clasify from model
            # save clasify result to folder outputs
            # return outputs path to url
            data = {'urlInputs': 'http://localhost:5000/uploads/inputs/'+filename,
                    'urlOutputs': 'http://localhost:5000/uploads/inputs/'+filename}
            return data


@app.route('/uploads/inputs/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_INPUT'],
                               filename,)


@app.route('/uploads/outputs/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_INPUT'],
                               filename,)
