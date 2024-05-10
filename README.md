# Whisper ASR Model for YouTube Videos
This project demonstrates how to use OpenAI's ASR (Automatic Speech Recognition) model "Whisper" to transcribe audio from YouTube videos.

## Overview

The Whisper model is a state-of-the-art ASR model developed by OpenAI, designed to transcribe speech accurately and efficiently. This project focuses on utilizing Whisper to transcribe speech from YouTube videos.

## Setup
1. Please follow the openai whisper guide on how to setup required materials.
https://github.com/openai/whisper/tree/main
2. Install other python dependencies.
    ```sh 
    pip install rich
    pip install yt_dlp
    ```

## Usage
```sh
CUDA_VISIBLE_DEVICES=0, python youtube_asr.py
CUDA_VISIBLE_DEVICES=0, python youtube_asr.py --youtube_link=[Your YouTube Link]
```