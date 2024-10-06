#!/bin/bash

for i in $(seq 0 11)
do
    screen -dmS "h0mc${i}" bash -c "source /cosma/local/anaconda3/5.2.0/etc/profile.d/conda.sh; source batch_h0mc_${i}.sh"
done

screen -list