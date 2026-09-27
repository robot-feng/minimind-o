import torch


def format_visual_prompt(image_tokens, text_prompt=""):
    image_tokens, text_prompt = image_tokens.strip(), text_prompt.strip()
    return f"{image_tokens}\n\n{text_prompt}" if text_prompt else image_tokens


def prepare_image_inputs(image, vision_processor, config, device="cpu"):
    pixels = vision_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
    prompt = config.image_special_token * config.image_token_len
    return {"pixel_values": pixels}, prompt
