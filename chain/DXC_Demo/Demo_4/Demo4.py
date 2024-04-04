from transformers import pipeline
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

oracle = pipeline(model="dandelin/vilt-b32-finetuned-vqa", token='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')
image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/lena.png"
pipe = pipeline("text-classification", token='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')
oracleItem=pipe(oracle(question="What is she wearing ?", image=image_url))

print(oracleItem)