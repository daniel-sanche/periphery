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

def CONFIDENCE_THRESHOLD():
    """
    Ignore matches with less than this score
    """
    var_name = inspect.stack()[0][3]
    return float(environ.get(var_name, 0.6))


def OUTPUT_BOXES():
    """
    Include bounding boxes in output
    """
    var_name = inspect.stack()[0][3]
    return environ.get(var_name, 'False') == 'True'


def OUTPUT_MASKS():
    """
    Include sementic masks in output
    """
    var_name = inspect.stack()[0][3]
    return environ.get(var_name, 'True') == 'True'
