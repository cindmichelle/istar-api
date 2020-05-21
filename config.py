from mrcnn.config import Config

UPLOAD_FOLDER_INPUT = './uploads/inputs'
UPLOAD_FOLDER_OUTPUT = './uploads/outputs'

class IstarConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    GPU_COUNT = 1
    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 5  # Background + 5 istar elements + 2 istar actor types

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 500
    BATCH_SIZE=1

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

     # use small validation steps since the epoch is small
    # VALIDATION_STEPS = 5

class InferenceConfig(IstarConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()
