# FastUltraVideo - Video Generation

This project uses infer.py to generate videos from text prompts or images using the UltraWAN model.

## Requirements
- Python 3.8+
- PyTorch
- Other dependencies listed in your environment (see imports in infer.py)

## Quickstart

1. Refer to [DiffSynth-Studio/examples/wanvideo](https://github.com/modelscope/DiffSynth-Studio/tree/main/examples/wanvideo) for environment preparation.
``` sh
pip install diffsynth==1.1.7
```
2. Download [Wan2.1-T2V-1.3B](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) model using huggingface-cli:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download --repo-type model Wan-AI/Wan2.1-T2V-1.3B --local-dir ultrawan_weights/Wan2.1-T2V-1.3B --resume-download
```
3. Download [UltraWan-1K/4K](https://huggingface.co/APRIL-AIGC/UltraWan) models using huggingface-cli:
``` sh
huggingface-cli download --repo-type model APRIL-AIGC/UltraWan --local-dir ultrawan_weights/UltraWan --resume-download
```
4. Generate native 1K/4K videos.
``` sh
==> one GPU
LoRA_1k: CUDA_VISIBLE_DEVICES=0 python infer.py --model_dir ultrawan_weights/Wan2.1-T2V-1.3B --model_path ultrawan_weights/UltraWan/ultrawan-1k.ckpt --mode lora --lora_alpha 0.25 --usp 0 --height 1088 --width 1920 --num_frames 81 --out_dir output/ultrawan-1k --image_path ./test.jpg --prompt "your prompt"
LoRA_4k: CUDA_VISIBLE_DEVICES=0 python infer.py --model_dir ultrawan_weights/Wan2.1-T2V-1.3B --model_path ultrawan_weights/UltraWan/ultrawan-4k.ckpt --mode lora --lora_alpha 0.5 --usp 0 --height 2160 --width 3840 --num_frames 33 --out_dir output/ultrawan-4k --image_path ./test.jpg --prompt "your prompt"
```
``` sh
==> usp with 6 GPUs
LoRA_1k: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 infer.py --model_dir ultrawan_weights/Wan2.1-T2V-1.3B --model_path ultrawan_weights/UltraWan/ultrawan-1k.ckpt --mode lora --lora_alpha 0.25 --usp 1 --height 1088 --width 1920 --num_frames 81 --out_dir output/ultrawan-1k --image_path ./test.jpg --prompt "your prompt"
LoRA_4k: CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 torchrun --standalone --nproc_per_node=6 infer.py --model_dir ultrawan_weights/Wan2.1-T2V-1.3B --model_path ultrawan_weights/UltraWan/ultrawan-4k.ckpt --mode lora --lora_alpha 0.5 --usp 1 --height 2160 --width 3840 --num_frames 33 --out_dir output/ultrawan-4k --image_path ./test.jpg --prompt "your prompt"
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