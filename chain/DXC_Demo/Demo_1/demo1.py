from transformers import pipeline
import os
os.environ['CURL_CA_BUNDLE'] = ''

pipe = pipeline("text-classification",
                token='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',
                trust_remote_code=True,
                device_map="auto")
pipe("This restaurant is awesome")