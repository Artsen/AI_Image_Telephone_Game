# Image Telephone

**Iteratively recreate an image through the “telephone game” of the OpenAI Images API.**

![Demo GIF or Screenshot](path/to/screenshot.gif)

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Installation](#installation)
* [Usage](#usage)
* [File Output](#file-output)
* [Logging & Debugging](#logging--debugging)
* [Configuration](#configuration)

## Overview

Image Telephone is a Streamlit app that plays the classic “telephone game” with images. You upload a source picture and the app repeatedly sends it through the OpenAI Images API using a full-image mask, aiming to produce an exact copy every iteration. Each output is fed back in as the next input for a fixed number of passes, and you can preview or export the results as a video.

## Features

* **Exact Replication:** Uses a full-image white mask to preserve colors, tinting, and details.
* **Custom Prompt:** Modify the default prompt to experiment with different instructions or leave it as the perfect-copy directive.
* **Retry Logic:** Automatically retries on SSL or network failures with exponential backoff.
* **Local Saving:** Saves every generated frame under `output/images/` and final videos under `output/videos/` with unique filenames.
* **Interactive Preview:** Browse through any generated frame via a slider.
* **Video Export:** Stitch all frames into an MP4 using `ffmpeg` with a single click.
* **Detailed Logs:** Collapsible log panel tracks every API call, retry, and save operation.

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/image-telephone.git
   cd image-telephone
   ```

2. **Create & activate a Python virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install `ffmpeg`** (required for video export)

   * macOS: `brew install ffmpeg`
   * Ubuntu: `sudo apt install ffmpeg`
   * Windows: Download from [ffmpeg.org](https://ffmpeg.org/) and add to PATH.

## Usage

1. **Run the app**

   ```bash
   streamlit run app.py
   ```
2. **In the sidebar:**

   * Enter your OpenAI API key.
   * (Optional) Customize the edit prompt.
   * Choose the number of iterations (passes).
3. **Upload an image** to start.
4. **Click ▶️ Start Iterations** to generate frames.
5. **Preview** any frame using the slider.
6. **Click 🎬 Generate Video** to export an MP4.

## File Output

* **Frames:** `output/images/frame_<iteration>_<uid>.png`
* **Videos:** `output/videos/video_<timestamp>.mp4`

All filenames include unique identifiers to avoid collisions.

## Logging & Debugging

Expand the **View Logs** section at the bottom of the app to see:

* API call attempts and retry messages.
* Success and error messages for each frame.
* File save paths for frames and videos.

## Configuration

* **DEFAULT\_PROMPT:** The default textual instruction for the OpenAI API.
* **FPS:** Frames per second for exported video.
* **IMAGE\_MODEL:** DALL·E model (`gpt-image-1` or `dall-e-2`).
* **MAX\_RETRIES:** How many times to retry on network errors.
* **ALLOWED\_SIZES:** Supported image dimensions for edits.
