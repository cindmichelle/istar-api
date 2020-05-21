import os
import skimage
import numpy as np

from config import UPLOAD_FOLDER_INPUT
from config import UPLOAD_FOLDER_OUTPUT

def detect_and_color_splash(model, image_path=''):
    assert image_path

    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(image_path))
        # Read image
        image = skimage.io.imread(os.path.join(UPLOAD_FOLDER_INPUT,image_path))
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{%S}".format(image_path)
        skimage.io.imsave(os.path.join(UPLOAD_FOLDER_OUTPUT,file_name), splash)

    print("Saved to ", file_name)
    return file_name

def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash
