#!/bin/bash

source ~/dotfiles/commands.sh

OPT=sgd
ACT=linear
MB=100
NL=4
LR=0.01
WD=0

EXEC=/home/gmoshkin/cmc/dl/autoencoder/entry.py

export INTEGRAM_TOKEN=cgdLBc9gt5e

time_start=$(date)

integram "started $(date)"

for ACT in linear relu sigmoid; do
  for OPT in sgd adam rms; do
    for MB in 50 100 ; do
      for NL in 2 4 6; do
        for LR in 0.1 0.01 0.001 0.0001; do
          for WD in 0 0.1 0.2 0.4 0.8; do
            old_dir="$(pwd)"
            dir="vis_${ACT}_${OPT}_MB${MB}_NL${NL}_LR${LR}_WD${WD}"
            date
            echo "cd to '$dir'"
            mkdir -p "$dir"
            cd "$dir"
            for i in 1 2 3 4; do
              "$EXEC" "MINIBATCH_SIZE=$MB" "N_LAYERS=$NL" "ACTIVATIONS=$ACT"\
                "OPTIM=$OPT" "WEIGHT_DECAY=$WD" "STEP_SIZE=$LR"
            done
            echo "cd back to '$old_dir'"
            cd "$old_dir"
          done
        done
      done
    done
  done
done

integram "ended $(date)"

echo "started $time_start"
echo "ended $(date)"
