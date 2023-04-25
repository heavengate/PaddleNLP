# export CUDA_VISIBLE_DEVICES=7
# python3.7 predict_generation.py --model=7b1_fp16 2>&1 | tee orig.txt

export CUDA_VISIBLE_DEVICES=4,5,6,7
python3.7 -m paddle.distributed.launch --gpus "4,5,6,7" predict_generation.py --model=mp4_7b1_fp16 2>&1 | tee orig.txt
