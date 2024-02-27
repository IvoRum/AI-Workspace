from transformers import pipeline
import ssl
import urllib.request

# Disable SSL certificate verification
ssl_context = ssl._create_unverified_context()

classifier= pipeline('sentiment-analysis',token='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')

pos='I love dogs'
neg='I really dog'

res=classifier(pos)

print(res)