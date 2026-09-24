import os

import cv2
import numpy as np
import torch
from PIL import Image


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


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
    pixel_values = torch.stack([
        vision_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        for image, _ in frames
    ]).unsqueeze(0).to(device)
    image_tokens = config.image_special_token * config.image_token_len
    prompt = "\n\n".join(
        f"Frame {index} at {timestamp:.2f}s:\n{image_tokens}"
        for index, (_, timestamp) in enumerate(frames, start=1)
    )
    return {"pixel_values": pixel_values}, prompt
