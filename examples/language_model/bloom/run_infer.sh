# export FLAGS_use_cublaslt_attn_gemm=0
# export FLAGS_use_cutlass_fmha=0
# export FLAGS_gemm_use_half_precision_compute_type=False
# export NVIDIA_TF32_OVERRIDE=0

export CUDA_VISIBLE_DEVICES=4,5,6,7
python3.7 -m paddle.distributed.launch --gpus "4,5,6,7" \
    infer_generation.py \
      --model_dir=./inference \
      --model_prefix=bloom \
      2>&1 | tee ep.txt
