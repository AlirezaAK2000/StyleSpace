#!/bin/bash


declare -a attributes=("bags-under-eyes" "big-nose" "high-cheekbones" "young" "arched-eyebrows" "brown-hair" "big-lips" "bushy-eyebrows" "chubby" "straight-hair")

for i in "${attributes[@]}"
do 
    echo "working on attribute $i"

    python MAdvance.py --dataset_name "celeba-hq_ffhq" \
    --model_name ffhq --bname "$i" \
    --positive_bank 500 --num_pos 30 \
    --start_index 20050 --exempelar_set "celeba-hq_ffhq" \
    --save_selected true

    python MAdvance.py --dataset_name "ffhq" \
    --model_name ffhq --bname "$i" \
    --positive_bank 500 --num_pos 30 \
    --start_index 0 --exempelar_set "ffhq" \
    --save_selected true
done 