import os
import skimage
import numpy as np

from config import UPLOAD_FOLDER_INPUT
from config import UPLOAD_FOLDER_OUTPUT
from mrcnn import visualize

def detect_and_color_splash(model, image_path=''):
    assert image_path

    if image_path:
        class_names = ['BG', 'Actor', 'Goal', 'Quality', 'Task', 'Resource']

        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))

        # Read image
        image = skimage.io.imread(os.path.join(UPLOAD_FOLDER_INPUT,image_path))

        # Detect objects
        r = model.detect([image], verbose=1)[0]

        # print("r['class_ids']", r['class_ids'])
        filename = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        class_names, r['scores'], filename=image_path)


    print("Saved to ", filename)
    return filename
