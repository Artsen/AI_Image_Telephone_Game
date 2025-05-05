#!/usr/bin/env python3
"""
Image Telephone: Iteratively recreate an image via OpenAI's Image Edit API.

This module defines:
- recreate_frame(): Calls the OpenAI edits endpoint with retry logic and saves each frame.
- create_video_ffmpeg(): Stitches saved frames into an MP4 video using ffmpeg.
- main(): Streamlit app entry point for user interaction.

Global Constants:
- DEFAULT_PROMPT (str): Default text prompt for perfect image replication.
- FPS (int): Frames per second for exported video.
- IMAGE_MODEL (str): OpenAI image model identifier.
- ALLOWED_SIZES (set[str]): Supported dimensions for edit requests.
- MAX_RETRIES (int): How many times to retry network calls.
- RETRY_DELAY (int): Delay in seconds between retry attempts.
- OUTPUT_DIR, IMAGES_DIR, VIDEOS_DIR (str): Paths for saving outputs.
"""
import os
import io
import time
import base64
import tempfile
import subprocess
import uuid

import streamlit as st
import openai
from PIL import Image
import requests

# --------------------------------------------------
# Configuration Constants (Global)
# --------------------------------------------------
DEFAULT_PROMPT = (
    "recreate this image without changing anything at all about the image, "
    "make sure to use the same colors and tinting, without adding any filters "
    "or sepia tones unless already there in the original image"
)
FPS = 6
IMAGE_MODEL = "gpt-image-1"
ALLOWED_SIZES = {"1024x1024", "1024x1536", "1536x1024"}
MAX_RETRIES = 5
RETRY_DELAY = 1  # seconds

# Directory setup for saved assets
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
VIDEOS_DIR = os.path.join(OUTPUT_DIR, "videos")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# --------------------------------------------------
# Function Definitions
# --------------------------------------------------

def recreate_frame(
    last_frame: Image.Image,
    prompt: str,
    iteration: int,
    model: str = IMAGE_MODEL
) -> Image.Image:
    """
    Send an image through the OpenAI Image Edit API to produce an exact replica.

    Performs up to MAX_RETRIES attempts on network or SSL errors, with RETRY_DELAY
    seconds between attempts. Saves each successfully generated frame locally.

    Args:
        last_frame (PIL.Image.Image): The source image to edit for this iteration.
        prompt (str): The textual instruction guiding the edit.
        iteration (int): Sequential index for naming the saved frame file.
        model (str): The OpenAI image model to use (e.g., 'gpt-image-1').

    Returns:
        PIL.Image.Image: The newly generated image frame. If editing fails,
        returns the original `last_frame` without modification.

    Raises:
        RuntimeError: If all retry attempts fail due to network issues.
    """
    logs = st.session_state.logs
    # Ensure frame mode compatibility: convert to RGBA
    last_frame = last_frame.convert("RGBA")

    # Determine image size parameter
    width, height = last_frame.size
    raw_size = f"{width}x{height}"
    size = raw_size if raw_size in ALLOWED_SIZES else "auto"

    # Create a full-white RGBA mask matching source dimensions
    mask = Image.new("RGBA", (width, height), (255, 255, 255, 255))

    # Serialize source image and mask to in-memory buffers
    img_buf = io.BytesIO()
    last_frame.save(img_buf, format="PNG")
    img_buf.name = "image.png"
    img_buf.seek(0)

    mask_buf = io.BytesIO()
    mask.save(mask_buf, format="PNG")
    mask_buf.name = "mask.png"
    mask_buf.seek(0)

    # Prepare API request parameters
    url = "https://api.openai.com/v1/images/edits"
    headers = {"Authorization": f"Bearer {openai.api_key}"}
    files = {
        "image": (img_buf.name, img_buf, "image/png"),
        "mask":  (mask_buf.name, mask_buf,  "image/png"),
    }
    data = {"model": model, "prompt": prompt, "n": 1, "size": size}

    # Attempt network call with retry logic
        # Retry loop with HTTP error handling and exponential backoff
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, files=files, data=data, timeout=30)
            resp.raise_for_status()
            break
        except requests.exceptions.SSLError as e:
            logs.append(f"Attempt {attempt}: SSL error ({e}), retrying...")
        except requests.exceptions.HTTPError as e:
            code = e.response.status_code
            err_msg = e.response.json().get("error", {}).get("message", e.response.text)
            if 400 <= code < 500:
                # Client-side error: abort further attempts
                logs.append(f"Client error {code}: {err_msg}")
                st.error(f"Edit failed: {err_msg}")
                return last_frame
            # Server-side error: log and retry
            logs.append(f"Server error {code}: {err_msg}, retrying...")
        except requests.exceptions.RequestException as e:
            logs.append(f"Attempt {attempt}: Network error ({e}), retrying...")
        # Exponential backoff delay
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY * (2 ** (attempt - 1)))
    else:
        logs.append("All retries failed.")
        st.error("Failed to fetch edit from OpenAI after multiple attempts.")
        return last_frame

    # Parse API response payload
    entry = resp.json().get("data", [{}])[0]
    if "b64_json" in entry:
        img_bytes = base64.b64decode(entry["b64_json"])
    else:
        img_url = entry.get("url")
        if not img_url:
            logs.append("No image returned in response.")
            return last_frame
        img_resp = requests.get(img_url)
        img_resp.raise_for_status()
        img_bytes = img_resp.content

    # Load PIL image from bytes and save locally
    frame = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    uid = uuid.uuid4().hex[:8]
    filename = f"frame_{iteration:03d}_{uid}.png"
    frame_path = os.path.join(IMAGES_DIR, filename)
    frame.save(frame_path)
    logs.append(f"Saved frame to {frame_path}")

    return frame


def create_video_ffmpeg(
    frames: list[Image.Image],
    fps: int,
    out_path: str
) -> None:
    """
    Stitch a list of PIL image frames into an MP4 video using ffmpeg.

    Args:
        frames (list[PIL.Image.Image]): Ordered sequence of frames.
        fps (int): Desired frame rate for the output video.
        out_path (str): File path for the generated MP4 file.

    Returns:
        None
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, frame in enumerate(frames, start=1):
            path = os.path.join(tmpdir, f"frame_{idx:04d}.png")
            frame.save(path)
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(tmpdir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            out_path
        ]
        subprocess.run(cmd, check=True)


def main() -> None:
    """
    Entry point for the Streamlit application UI.

    - Collects user inputs (API key, prompt, iterations).
    - Uploads initial image.
    - Iteratively calls recreate_frame().
    - Provides frame preview and video export.

    Returns:
        None
    """
    st.set_page_config(page_title="Image Telephone", layout="wide")

    # Sidebar: API credentials and settings
    with st.sidebar:
        st.header("Settings")
        api_key = st.text_input("OpenAI API Key", type="password")
        if not api_key:
            st.warning("Please enter your OpenAI API key.")
            return
        openai.api_key = api_key

        prompt = st.text_area("Edit Prompt", value=DEFAULT_PROMPT, height=120)
        max_iter = st.number_input("Max iterations", min_value=1, max_value=200, value=5)

    st.title("🔄 Image Telephone")
    uploaded = st.file_uploader("Upload an image to start", type=["png", "jpg", "jpeg"])
    if not uploaded:
        st.info("Please upload an image to begin.")
        return

    # Load and display original
    original = Image.open(uploaded).convert("RGBA")
    st.image(original, caption="Original", width=300)

    # Initialize session state
    if "frames" not in st.session_state:
        st.session_state.frames = [original]
        st.session_state.started = False
        st.session_state.done = False
        st.session_state.logs = []

    # Start iterations button
    if not st.session_state.started:
        if st.button("▶️ Start Iterations"):
            st.session_state.started = True
    # Perform iterative editing
    elif not st.session_state.done:
        prog = st.progress(0)
        cur = st.empty()
        thumbs = st.empty()
        for i in range(1, max_iter + 1):
            frame = recreate_frame(st.session_state.frames[-1], prompt, i)
            st.session_state.frames.append(frame)
            prog.progress(i / max_iter)
            cur.image(frame, caption=f"Iteration {i}", width=400)
            thumbs.image(st.session_state.frames, width=80)
        st.session_state.done = True

    # Preview frames and export video
    if st.session_state.done:
        st.subheader("Preview Frames")
        idx = st.slider("Select frame to preview", 0, len(st.session_state.frames) - 1, 0)
        st.image(st.session_state.frames[idx], caption=f"Frame {idx}", width=400)

        if st.button("🎬 Export Video"):
            out_file = os.path.join(VIDEOS_DIR, f"video_{int(time.time())}.mp4")
            create_video_ffmpeg(st.session_state.frames, FPS, out_file)
            st.video(out_file)

        with st.expander("Logs"):
            for entry in st.session_state.logs:
                st.write(entry)


if __name__ == "__main__":
    main()
