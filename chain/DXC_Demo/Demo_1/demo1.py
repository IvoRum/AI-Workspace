from transformers import pipeline
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


pipe = pipeline("text-classification", token='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')
print(pipe("This restaurant is awesome"))