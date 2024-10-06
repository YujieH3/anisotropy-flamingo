#!/bin/bash

for i in $(seq 0 11)
do
    screen -dmS "bfmc${i}" bash -c "source /cosma/local/anaconda3/5.2.0/etc/profile.d/conda.sh; source batch_bfmc${i}.sh"
done

screen -list