# export FLAGS_use_cublaslt_attn_gemm=0
# export FLAGS_use_cutlass_fmha=0
# export FLAGS_gemm_use_half_precision_compute_type=False
# export NVIDIA_TF32_OVERRIDE=0

# python3.7 fuse_mt_generation.py --model=20000 2>&1 | tee fuse.txt

export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_ALGO=Tree
python3.7 -m paddle.distributed.launch --gpus "4,5,6,7" fuse_mt_generation.py --model=mp4_7b1 2>&1 | tee mt.txt
