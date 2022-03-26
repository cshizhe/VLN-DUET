NODE_RANK=0
NUM_GPUS=1
outdir=../datasets/SOON/exprs_map/pretrain/cmt-vitbase.butdobj-mlm.sap.og-init.lxmert

# train
CUDA_VISIBLE_DEVICES='0' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    train_soon_obj.py --world_size ${NUM_GPUS} \
    --vlnbert cmt \
    --model_config config/soon_obj_model_config.json \
    --config config/soon_obj_pretrain.json \
    --output_dir $outdir