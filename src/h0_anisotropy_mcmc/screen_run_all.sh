




for i in $(seq 0 11)
do
    screen -S "h0mc${i}"

    # run h0mc
    source batch_h0mc_${i}.sh

    # detach
    screen -d

done