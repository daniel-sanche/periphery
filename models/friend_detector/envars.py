from os import environ
import inspect


def CONTROLLER_ADDRESS():
    """
    The address of the controller server
    """
    var_name = inspect.stack()[0][3]
    return environ.get(var_name, 'localhost:8080')


def INACTIVITY_THRESHOLD():
    """
    How long (in seconds) to wait for a response before stating to poll
    """
    var_name = inspect.stack()[0][3]
    return int(environ.get(var_name, 5))


def POLL_TIME():
    """
    How often to send new frame requests when controller isn't responding
    """
    var_name = inspect.stack()[0][3]
    return int(environ.get(var_name, 1))


def AUTO_RUN():
    """
    Sets whether the model should request images from the controller
    """
    var_name = inspect.stack()[0][3]
    return environ.get(var_name, 'True') == 'True'


#############################################

def RECOGNITION_CONFIDENCE_THRESHOLD():
    """
    Ignore label recognitions with less than this score
    """
    var_name = inspect.stack()[0][3]
    return float(environ.get(var_name, 0.9))

def DETECTION_CONFIDENCE_THRESHOLD():
    """
    Ignore potential haar faces with less than this score
    """
    var_name = inspect.stack()[0][3]
    return int(environ.get(var_name, 20))

def UNKNOWN_LABEL():
    """
    The label given to unknown faces
    """
    var_name = inspect.stack()[0][3]
    return environ.get(var_name, '???')

def USE_IMAGE_DATASET():
    """
    Sets whether to load local images to the dataset
    """
    var_name = inspect.stack()[0][3]
    return environ.get(var_name, 'True') == 'True'

def USE_PICKLE_DATASET():
    """
    Sets whether to load a pickle file to the dataset
    """
    var_name = inspect.stack()[0][3]
    return environ.get(var_name, 'True') == 'True'

def SAVE_DATASET_TO_PICKLE():
    """
    Sets whether to save data to a pickle file after loading
    """
    var_name = inspect.stack()[0][3]
    return environ.get(var_name, 'False') == 'True'
