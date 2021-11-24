NODE_RANK=0
NUM_GPUS=2
outdir=../datasets/REVERIE/exprs_map/pretrain/cmt-vitbase-mlm.mrc.sap.og-init.lxmert-aug.speaker

# train
CUDA_VISIBLE_DEVICES='0,1' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    train_reverie_obj.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/reverie_obj_model_config.json \
    --config config/reverie_obj_pretrain.json \
    --output_dir $outdir

