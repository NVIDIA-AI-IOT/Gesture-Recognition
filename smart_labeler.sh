# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

xterm -e "roscore" &
# $1 should either be empty or --camera=int
# $2 should be -o
if [ -z $3 ]; then
    NUM_FRAMES=4
else
    NUM_FRAMES=$3
fi

xterm -e "python lstm_wave_inference.py $1 $2 -f=$NUM_FRAMES" &
sleep 20
xterm -e "python lstm_inference.py" &
sleep 20
python smart_labeler.py $2
