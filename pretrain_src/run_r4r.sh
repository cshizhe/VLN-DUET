NODE_RANK=0
NUM_GPUS=1
outdir=../datasets/R4R/exprs_map/pretrain/cmt-vitbase-mlm.sap-init.lxmert

# train
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    train_r4r.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/r2r_model_config.json \
    --config config/r4r_pretrain.json \
    --output_dir $outdir
