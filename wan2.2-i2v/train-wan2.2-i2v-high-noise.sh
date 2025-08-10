#!/bin/bash
set -euo pipefail

source "$MUSUBI_SCRIPTS_DIR/env.sh"

accelerate launch --num_cpu_threads_per_process 1 "$MUSUBI_HOME/src/musubi_tuner/wan_train_network.py" \
    --task i2v-A14B \
    --dit "$BASE_DIR/models/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors" \
    --vae "$BASE_DIR/models/vae/wan_2.1_vae.safetensors" \
    --t5 "$BASE_DIR/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
    --dataset_config "$BASE_DIR/dataset/dataset.toml" \
    --mixed_precision fp16 \
    --fp8_base \
    --optimizer_type adamw \
    --learning_rate 3e-4 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 1 \
    --max_data_loader_n_workers 2 \
    --network_module networks.lora_wan \
    --network_dim 16 \
    --network_alpha 16 \
    --timestep_sampling shift \
    --discrete_flow_shift 1.0 \
    --max_train_epochs 100 \
    --save_every_n_epochs 100 \
    --seed 5 \
    --optimizer_args weight_decay=0.1 \
    --max_grad_norm 0 \
    --lr_scheduler polynomial \
    --lr_scheduler_power 8 \
    --lr_scheduler_min_lr_ratio="5e-5" \
    --output_dir "$BASE_DIR/output" \
    --output_name WAN2.2-HighNoise_SmartphoneSnapshotPhotoReality_v3_by-AI_Characters \
    --metadata_title WAN2.2-HighNoise_SmartphoneSnapshotPhotoReality_v3_by-AI_Characters \
    --metadata_author AI_Characters \
    --preserve_distribution_shape \
    --min_timestep 875 \
    --max_timestep 1000 \
    --blocks_to_swap 1
