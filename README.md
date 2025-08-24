# QImg

QImg is a simple Flask web app to generate and edit images using Qwen/Qwen-Image (generation) and Qwen/Qwen-Image-Edit (editing) via Diffusers. It provides:

- Gallery as the homepage, listing images from `static/outputs/`
- Generate page: text-to-image
- Edit page: image edit from upload or selecting an existing gallery image
- Background jobs: generation and edits run in worker threads; the Gallery polls for completion and prepends new images
- Gallery actions: Download and Delete

## Requirements

- Python 3.10+
- A virtual environment (recommended)
- PyTorch appropriate for your platform (this README pins CPU/CUDA via `pip` automatically based on your wheel selection)

Core dependencies are pinned in `requirements.txt`:

- Flask, Pillow
- torch, torchvision, accelerate
- diffusers (git pin), transformers, tokenizers, safetensors
- huggingface-hub and small utilities (filelock, fsspec, tqdm, requests, pyyaml, regex)

## Models

This app uses the following Hugging Face models through the Diffusers library:

- Qwen/Qwen-Image — for text-to-image generation
- Qwen/Qwen-Image-Edit — for image editing with text prompts

Notes:

- Models are downloaded on first use and cached under `~/.cache/huggingface/`.
- Some models may require accepting licenses/terms on Hugging Face.
- CPU works but is slower; a CUDA GPU is recommended for performance.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you need CUDA-enabled PyTorch, install the matching wheels before or after the above step per: https://pytorch.org/get-started/locally/

## Run

```bash
. .venv/bin/activate
python main.py
# App runs at http://127.0.0.1:5000
```

The app will create these directories as needed:

- `uploads/` for uploaded images
- `static/outputs/` for generated and edited results

## Usage

- Open the app homepage (Gallery)
- Generate: fill prompt on Generate page and submit
- Edit: either upload an image or navigate from Gallery using "Edit this" on a card
- While jobs are running, you’ll be redirected to the Gallery; when they finish, a banner appears and new items show at the top

### Gallery actions

- Download: saves the image to your machine
- Delete: removes the image file from `static/outputs/` and the card from the UI
