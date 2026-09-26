import os

import cv2
import numpy as np
import torch
from PIL import Image


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
DEFAULT_VIDEO_FRAMES = 4


def repeat_static_image_frames(pixel_values, num_frames=DEFAULT_VIDEO_FRAMES):
    """Expand a batch of still images to the shared video-frame layout."""
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if pixel_values.ndim != 4:
        raise ValueError("pixel_values must have shape (batch, channels, height, width)")
    return pixel_values.unsqueeze(1).expand(-1, num_frames, -1, -1, -1)


def format_frame_prompt(image_tokens, timestamps):
    return "\n\n".join(
        f"Frame {index} at {timestamp:.2f}s:\n{image_tokens}"
        for index, timestamp in enumerate(timestamps, start=1)
    )


def format_static_image_prompt(image_tokens, num_frames=DEFAULT_VIDEO_FRAMES):
    """Give a still image one visual token block for every shared frame slot."""
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    return "\n\n".join([image_tokens] * num_frames)


def sample_video_frames(video_path, num_frames=4):
    if num_frames < 1:
        raise ValueError("num_frames must be positive")

    capture = cv2.VideoCapture(os.fspath(video_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < 1:
            raise ValueError(f"Video has no readable frames: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        indices = np.linspace(0, frame_count - 1, min(num_frames, frame_count)).round().astype(int)
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append((Image.fromarray(rgb), float(index / fps) if fps > 0 else 0.0))
        if not frames:
            raise ValueError(f"Could not decode frames from video: {video_path}")
        return frames
    finally:
        capture.release()


def prepare_video_inputs(video_path, vision_processor, config, device="cpu", num_frames=4):
    frames = sample_video_frames(video_path, num_frames=num_frames)
    if len(frames) < num_frames:
        frames.extend([frames[-1]] * (num_frames - len(frames)))
    pixel_values = torch.stack([
        vision_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        for image, _ in frames
    ]).unsqueeze(0).to(device)
    image_tokens = config.image_special_token * config.image_token_len
    prompt = format_frame_prompt(image_tokens, [timestamp for _, timestamp in frames])
    return {"pixel_values": pixel_values}, prompt


def prepare_image_inputs(image, vision_processor, config, device="cpu", num_frames=DEFAULT_VIDEO_FRAMES):
    """Repeat one still image into the same frame tensor shape used by videos."""
    pixels = vision_processor(images=image, return_tensors="pt")["pixel_values"]
    frame_batch = repeat_static_image_frames(pixels, num_frames=num_frames).to(device)
    static_mask = torch.ones(frame_batch.size(0), dtype=torch.bool, device=device)
    image_tokens = config.image_special_token * config.image_token_len
    prompt = format_static_image_prompt(image_tokens, num_frames)
    return {"pixel_values": frame_batch, "static_image_mask": static_mask}, prompt
