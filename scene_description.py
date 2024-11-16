from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image
import numpy as np
import torch

def generate_short_captions(image_path, num_captions=3):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    raw_image = Image.open(image_path).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(
        pixel_values=inputs['pixel_values'],
        max_length=50,
        num_return_sequences=num_captions,  
        do_sample=True,
        temperature=0.7,
    )

    short_captions = [processor.decode(out[i], skip_special_tokens=True) for i in range(num_captions)]
    return short_captions

def expand_description(captions):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    combined_captions = " | ".join(captions)
    prompt = (
        f"Write a short story based on these scenes: {combined_captions}. "
        "Describe the setting in a way that creates a vivid scene, aiming for 2-3 sentences."
    )

    inputs = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=150,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )

    expanded_description = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean up the output
    if prompt in expanded_description:
        expanded_description = expanded_description.replace(prompt, "").strip()

    return expanded_description

def generate_scene_description(image_path):
    short_captions = generate_short_captions(image_path, num_captions=3)  # Specify 3 captions
    expanded_description = expand_description(short_captions)

    return expanded_description 
