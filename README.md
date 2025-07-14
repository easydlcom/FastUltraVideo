# FastUltraVideo - Video Generation

This project uses infer.py to generate videos from text prompts or images using the UltraWAN model.

## Requirements
- Python 3.8+
- PyTorch
- Other dependencies listed in your environment (see imports in infer.py)

## Usage


Image + Prompt:
```bash
python infer.py --model_dir ultrawan_weights/Wan2.1-T2V-1.3B --mode full --height 1088 --width 1920 --num_frames 81 --out_dir output/ori --image_path ./test.jpg --prompt "your prompt"
```

Image only:
```bash
python infer.py --model_dir ultrawan_weights/Wan2.1-T2V-1.3B --mode full --height 1088 --width 1920 --num_frames 81 --out_dir output/ori --image_path ./test.jpg
```

Prompt only:
```bash
python infer.py --model_dir ultrawan_weights/Wan2.1-T2V-1.3B --mode full --height 1088 --width 1920 --num_frames 81 --out_dir output/ori --prompt "your prompt"
```

## Arguments
- `--model_dir`: Path to model weights directory (default: ultrawan_weights/Wan2.1-T2V-1.3B)
- `--model_path`: Path to specific model file (optional)
- `--mode`: Model mode, e.g., 'full' or 'lora' (default: full)
- `--lora_alpha`: LoRA alpha value (default: 1.0)
- `--usp`: Use distributed USP (default: 0)
- `--height`: Video frame height (default: 480)
- `--width`: Video frame width (default: 832)
- `--num_frames`: Number of frames in output video (default: 81)
- `--out_dir`: Output directory (default: output/ori)
- `--image_path`: Path to input image for image2video
- `--prompt`: Text prompt for video generation

## Output
Videos are saved in the specified output directory as MP4 files.

## Example
```bash
python infer.py --prompt "A lively puppy runs on a green lawn." --height 480 --width 832 --num_frames 81
```

## Notes
- If no prompt is provided, two default prompts are used.
- If an image is provided, image-to-video mode is used.

## License
Apache-2.0 license