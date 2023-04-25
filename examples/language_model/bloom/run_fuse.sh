# export CUDA_VISIBLE_DEVICES=7
# python3.7 fuse_mt_generation.py --model=7b1_fp16 --fp16 2>&1 | tee mt.txt

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_ALGO=Tree
python3.7 -m paddle.distributed.launch --gpus "0,1,2,3" fuse_mt_generation.py --model=mp4_7b1_fp16 --fp16 2>&1 | tee mt.txt
