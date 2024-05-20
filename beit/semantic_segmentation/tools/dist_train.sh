#!/usr/bin/env bash
export CUDA_HOME='/opt/cuda-10.2'
module load gpu/cuda-10.2
export PYTHONPATH="/misc/home6/s0107/unilm/beit/semantic_segmentation/:$PYTHONPATH"


# ПРИМЕЧАНИЕ: после слова wrap вставлять все строки в одинарных кавычках
sbatch \
--cpus-per-task=9 \
--mem=32000 \
-p v100 --gres=gpu:v100:2 \
-t 10:00:00 \
--job-name=segm \
--output=./logs/"%j" \
\
--wrap="python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=-29500 train.py \
/misc/home6/s0107/unilm/beit/semantic_segmentation/configs/beit/upernet/upernet_beit_base_12_640_slide_160k_ade20k_pt2ft.py \
--gpus 2 \
--work-dir /misc/home6/s0107/unilm/beit/semantic_segmentation/tools/logs/ \
--seed 0  \
--deterministic \
--options model.pretrained='/misc/home6/s0107/unilm/beit/semantic_segmentation/tools/pretrained/beit_base_patch16_224_pt22k_ft22k.pth' \
--launcher pytorch \
"
# Если все узлы заняты и хочется запустить задачу на пол часа на отладочном узле на CPU,
#  вставить две следующие строки после слова sbatch и указать в начале этого скрипта работу на CPU
# -p debug \
# -t 00:30:00
