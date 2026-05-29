#!/bin/bash
# Run CaTSG experiments on 3 GPUs in parallel
# GPU 0: CaTSG AQ
# GPU 1: CaTSG Traffic
# GPU 2: T2S ETTh1

set -e

PROJECT_DIR="/home/newuser001/huangyu/Research/TSDIT"
DATA_DIR="${PROJECT_DIR}/Three Levels Data"
CONFIG_DIR="${PROJECT_DIR}/configs"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Starting CaTSG + ETTh1 experiments"
echo "=========================================="

# GPU 0: CaTSG AQ
CUDA_VISIBLE_DEVICES=0 python -m mmldm.tiger.train \
    --data_dir "${DATA_DIR}/CaTSG" \
    --config "${CONFIG_DIR}/catsg_aq.json" \
    --dataset_type csv \
    --datasets catsg_aq \
    --time_interval 96 \
    --epochs 500 \
    --batch_size 256 \
    --save_dir "${PROJECT_DIR}/checkpoints/catsg_aq" \
    --log_dir "${PROJECT_DIR}/runs/catsg_aq" \
    > "${LOG_DIR}/catsg_aq.log" 2>&1 &

PID_AQ=$!
echo "[GPU 0] CaTSG AQ training started (PID: $PID_AQ)"

# GPU 1: CaTSG Traffic
CUDA_VISIBLE_DEVICES=1 python -m mmldm.tiger.train \
    --data_dir "${DATA_DIR}/CaTSG" \
    --config "${CONFIG_DIR}/catsg_traffic.json" \
    --dataset_type csv \
    --datasets catsg_traffic \
    --time_interval 96 \
    --epochs 500 \
    --batch_size 256 \
    --save_dir "${PROJECT_DIR}/checkpoints/catsg_traffic" \
    --log_dir "${PROJECT_DIR}/runs/catsg_traffic" \
    > "${LOG_DIR}/catsg_traffic.log" 2>&1 &

PID_TRAFFIC=$!
echo "[GPU 1] CaTSG Traffic training started (PID: $PID_TRAFFIC)"

# GPU 2: T2S ETTh1 (text-only, time_interval=24)
CUDA_VISIBLE_DEVICES=2 python -m mmldm.tiger.train \
    --data_dir "${DATA_DIR}/TSFragment-600K" \
    --config "${CONFIG_DIR}/t2s_exp2_text_only.json" \
    --dataset_type csv \
    --datasets ETTh1 \
    --time_interval 24 \
    --epochs 500 \
    --batch_size 512 \
    --save_dir "${PROJECT_DIR}/checkpoints/etth1_24" \
    --log_dir "${PROJECT_DIR}/runs/etth1_24" \
    > "${LOG_DIR}/etth1_24.log" 2>&1 &

PID_ETT=$!
echo "[GPU 2] ETTh1 training started (PID: $PID_ETT)"

echo ""
echo "All 3 experiments launched:"
echo "  - CaTSG AQ:      GPU 0, PID $PID_AQ"
echo "  - CaTSG Traffic:  GPU 1, PID $PID_TRAFFIC"
echo "  - ETTh1:          GPU 2, PID $PID_ETT"
echo ""
echo "Logs: ${LOG_DIR}/"
echo "Checkpoints: ${PROJECT_DIR}/checkpoints/"
echo ""
echo "Monitor with: tail -f ${LOG_DIR}/*.log"
echo "=========================================="

# Wait for all background jobs
wait $PID_AQ $PID_TRAFFIC $PID_ETT
echo "All experiments completed!"
