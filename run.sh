CUDA_VISIBLE_DEVICES=$3 python -m torch.distributed.launch --master_port $4 main.py \
    --project_title $1_$2_$3_$4 \
    --method $2 \
    --dataset_type $1 \
	--e_lr 4e-5 \
	--r_lr 4e-5 \
	--online_tracker 1 \
	--tr_batch_size 128 \
	--model alexnet \
	--save_path /PATH/ \
	--print_check 100

CUDA_VISIBLE_DEVICES=$3 python -m torch.distributed.launch --master_port $4 main.py \
    --project_title $1_$2_$3_$4 \
    --method $2 \
    --dataset_type $1 \
	--e_lr 4e-5 \
	--r_lr 4e-5 \
	--online_tracker 1 \
	--tr_batch_size 128 \
	--model alexnet \
	--save_path /PATH/ \
	--print_check 100

CUDA_VISIBLE_DEVICES=$3 python -m torch.distributed.launch --master_port $4 main.py \
    --project_title $1_$2_$3_$4 \
    --method $2 \
    --dataset_type $1 \
	--e_lr 1e-4 \
	--r_lr 1e-4 \
	--no_domain 5 \
	--topk 5 \
	--domain_sampling none \
	--online_tracker 1 \
	--tr_batch_size 128 \
	--ermfc_input_dim 512 \
	--ermfc_output_dim 2 \
	--warmup_coef1 10 \
	--warmup_coef2 200 \
	--model alexnet \
	--save_path /PATH/ \
	--print_check 100
