#! /bin/bash

WORKING_DIR=.

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8

#HOST_FILE="${WORKING_DIR}/configs/host_files/hostfile-cpm2"

MP_SIZE=1

DATA_EXT=".jsonl"
DATA_PATH="./data/adgen"

LR=${1-0.00001}
GRAD_ACC=${2-1}
echo $LR
echo $GRAD_ACC
ENC_LEN=512
DEC_LEN=256
CONFIG_PATH="${WORKING_DIR}/configs/model/cpm2_config.json"
CKPT_PATH="${WORKING_DIR}/checkpoints/cpm2-small-blr"

SAVE_PATH="${WORKING_DIR}/results/adgen/cpm2_finetune_lr${LR}const_G${GRAD_ACC}/"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/configs/deepspeed/ds_full_model.json"
TOKENIZER_PATH="${WORKING_DIR}/bpe_cn"

BATCH_SIZE=64
EVAL_BATCH_SIZE=128
TRAIN_ITER=-1
EPOCHS=10

TOP_P=0.9
TOP_K=40


OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --eval-batch-size ${EVAL_BATCH_SIZE}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --enc-seq-length ${ENC_LEN}"
OPTS+=" --dec-seq-length ${DEC_LEN}"
OPTS+=" --train-iters ${TRAIN_ITER}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --data-ext ${DATA_EXT}"
OPTS+=" --data-name adgen"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style constant"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --warmup 0.0"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --save-interval 20000"
OPTS+=" --eval-interval 100"
OPTS+=" --eval-iters 10"
OPTS+=" --log-interval 10"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --do-train"
OPTS+=" --do-valid"
# OPTS+=" --do-eval"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --top-p ${TOP_P}"
OPTS+=" --top-k ${TOP_K}"
MASTER_PORT=6009
NODE_RANK=$SLURM_PROCID
MASTER_ADDR=`python ${WORKING_DIR}/myregex.py ${SLURM_STEP_NODELIST}`

CMD="python  -m torch.distributed.launch --nnodes ${NUM_WORKERS} --node_rank ${NODE_RANK} --nproc_per_node ${NUM_GPUS_PER_WORKER} --master_addr ${MASTER_ADDR}  --master_port ${MASTER_PORT}  ${WORKING_DIR}/finetune_cpm2.py  $@ ${OPTS}"

#CMD="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE} ${WORKING_DIR}/finetune_cpm2.py ${OPTS}"
echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
