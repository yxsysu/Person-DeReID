# train
# CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --use-env --nproc_per_node=2 --master_port 61234 train.py --config_file configs/TargetMSMT17/exp_1.yml MODEL.DIST_TRAIN True

# train
# python lora_finetune.py --config configs/TargetOccDuke/exp_34_pid_50.yml
# python lora_finetune.py --config configs/occ_duke_ab/pid_50_wo_style.yml
# python lora_finetune.py --config configs/occ_duke_ab/pid_25_wo_style.yml

python lora_finetune.py --config configs/DeReIDMSMT/exp_2.yml