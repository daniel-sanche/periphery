from base64 import b64decode, b64encode
from PIL import Image
import numpy as np
import io

def data_url_to_pil(data_url):
    image_data = b64decode(data_url.split(',')[1])
    img = Image.open(io.BytesIO(image_data))
    return img


def np_img_to_data_url(np_img, mode='RGB'):
    with io.BytesIO() as output:
        pil_img = Image.fromarray(np_img, mode=mode)
        pil_img.save(output, format="WebP")
        contents = output.getvalue()
        data64 = b64encode(contents)
        data_url = 'data:image/webp;base64,'+data64.decode('ascii')
        return data_url


