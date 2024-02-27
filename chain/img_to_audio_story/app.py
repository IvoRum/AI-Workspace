from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
import os
os.environ['CURL_CA_BUNDLE'] = ''

load_dotenv(find_dotenv())
def imgToTest(url):
    image_to_text= pipeline("image-to-text", model="Salesforce/blip-image-captioning-base",token='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')

    test=image_to_text(url)

    print(test)
    return test

imgToTest("testimg.jpg")
