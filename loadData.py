import urllib
from PIL import Image
import torchvision.datasets as datasets

class Data:
    def load_single_image(self, filename):
        input_image = Image.open(filename)
        return [input_image]
