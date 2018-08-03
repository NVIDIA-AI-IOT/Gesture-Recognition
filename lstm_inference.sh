# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
if [ -z $1 ]; then
    CAM_PORT=1
else
    CAM_PORT=$1
fi

if [ -z $2 ]; then
    NUM_FRAMES=4
else
    NUM_FRAMES=$2
fi

if [ -z $3 ]; then
    CKPT_FILE="lstm307.ckpt"
else
    CKPT_FILE=$3
fi



xterm -e "roscore" &
# $1 is camera port num
# $2 is an int
# $3 is a ckpt file
# $4 is -o 
# $5 is -a
# $6 is -m
xterm -e "python thread_inf.py $4 $5 --camera=$CAM_PORT -f=$NUM_FRAMES" &
sleep 20
python lstm_inference.py $4 $6 -c=$CKPT_FILE
