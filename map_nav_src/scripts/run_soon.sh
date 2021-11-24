
train_alg=dagger

features=vitbase
ft_dim=768
obj_features=butd
obj_ft_dim=2048

ngpus=1
seed=0

name=${train_alg}-${features}
name=${name}-seed.${seed} 

outdir=${DATA_ROOT}/SOON/exprs_map/finetune/${name}


flag="--root_dir ${DATA_ROOT}
      --dataset soon
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert

      --enc_full_graph
      --graph_sprels
      --fusion dynamic
      --multi_endpoints

      --dagger_sample sample

      --train_alg ${train_alg}
      
      --num_l_layers 9
      --num_x_layers 4
      --num_pano_layers 2
      
      --max_action_len 20
      --max_instr_len 100
      --max_objects 100

      --batch_size 2
      --lr 1e-5
      --iters 200000
      --log_every 1000
      --optim adamW

      --features ${features}
      --obj_features ${obj_features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4
      --obj_feat_size ${obj_ft_dim}

      --ml_weight 0.2   

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."

CUDA_VISIBLE_DEVICES='0' python soon/main.py $flag  \
      --tokenizer bert \
      --bert_ckpt_file '' \
      --eval_first

# test
CUDA_VISIBLE_DEVICES='0' python soon/main.py $flag  \
      --tokenizer bert \
      --resume_file ../datasets/SOON/trained_models/best_val_unseen_house \
      --test --submit