import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename

import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn import utils

from config import InferenceConfig
from dataset import IstarDataset
from detect import detect_and_color_splash
from config import UPLOAD_FOLDER_INPUT, UPLOAD_FOLDER_OUTPUT

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


ROOT_DIR = os.path.abspath("./")

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# # Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# # Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#     utils.download_trained_weights(COCO_MODEL_PATH)


app = Flask(__name__)
app.config['UPLOAD_FOLDER_INPUT'] = UPLOAD_FOLDER_INPUT
app.config['UPLOAD_FOLDER_OUTPUT'] = UPLOAD_FOLDER_OUTPUT

# model = model.load

# TODO :
# - copy inference class dan train config class (karena inference class inherit dr config class) v
# - model.py v
# - config.py v
# - utils.py v
# - visualize.py v

# - classify dr data static


# # Training dataset
# dataset_train = IstarDataset()
# dataset_train.load_istar("./dataset_05_10/", "train")
# dataset_train.prepare()

# # Validation dataset
# dataset_val = IstarDataset()
# dataset_val.load_istar("./dataset_05_10/", "val")
# dataset_val.prepare()

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)




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
            input_filename = secure_filename(file.filename)
            file.save(os.path.join(
                app.config['UPLOAD_FOLDER_INPUT'], input_filename))

            # call clasify from model
            # save clasify result to folder outputs
            # return outputs path to url

            output_filename = detect_and_color_splash(model, image_path=input_filename)

            data = {'urlInputs': 'http://localhost:5000/uploads/inputs/'+input_filename,
                    'urlOutputs': 'http://localhost:5000/uploads/outputs/'+output_filename}
            return jsonify(data)

# test if model has been succeed to classify
@app.route('/classify')
def classify_model():
    input_filename = '20200218_165149.jpg'
    output_filename = detect_and_color_splash(model, image_path=input_filename)

    data = {'urlInputs': 'http://localhost:5000/uploads/inputs/'+input_filename,
            'urlOutputs': 'http://localhost:5000/uploads/outputs/'+output_filename}

    return jsonify(data)



@app.route('/uploads/inputs/<filename>')
def uploaded_input_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_INPUT'],
                               filename,)

@app.route('/uploads/outputs/<filename>')
def uploaded_output_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_OUTPUT'],
                               filename,)
