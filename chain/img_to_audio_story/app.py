from dotenv import load_dotenv, find_detenv
from transformers import pipeline

load_dotenv(find_detenv())

def imgToTest(url):
    image_to_text= pipeline("image_to_text", model="Salesforce/blip-image-captioning-base")

    test=image_to_text(url)[0]["generated_text"]

    print(test)
    return test

imgToTest("testimg.jpg")
