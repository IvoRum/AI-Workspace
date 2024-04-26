from transformers import pipeline

oracle = pipeline(model="dandelin/vilt-b32-finetuned-vqa", token='TOKEN')
image_url = "https://huggingface.co/datasets/Narsil/image_dummy/raw/main/lena.png"
print(oracle(question="What is she wearing ?", image=image_url))
