from PIL import Image
import requests
from io import BytesIO

image_url = "https://cc0.photo/wp-content/uploads/2016/11/Fluffy-orange-cat-980x735.jpg"
Image.open(BytesIO(requests.get(image_url).content))

from transformers import pipeline

estimator = pipeline(task="depth-estimation", model="Intel/dpt-large", token='TOKEN')
result = estimator(images=image_url)
result

result['depth']
