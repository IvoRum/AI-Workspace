from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

def imgToTest(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base",
                             token='TOKEN')

    test = image_to_text(url)[0]["Generated_text"]

    print(test)
    return test


def generateStory(scenario):
    model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",token='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')

    messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant",
         "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
        {"role": "user", "content": "Do you have mayonnaise recipes?"}
    ]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

    outputs = model.generate(inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


generateStory(imgToTest("https://www.justrunlah.com/wp-content/uploads/2015/03/Running-sport-man.-Fit-muscular-young-male-runner-sprinting-at-great-speed-outdoors-on-road.-e1425896049776.jpg"))
