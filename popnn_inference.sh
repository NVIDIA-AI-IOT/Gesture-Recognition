# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

xterm -e "roscore" &
if [ -z $1 ]; then
  PORT="0"
else
  PORT=$1
fi

if [ -z $2 ]; then
  CKPT_FILE="popnn4311.ckpt"
else
  CKPT_FILE=$2
fi

if [ -z $3 ]; then
  F_BUF_SIZE="4"
else
  F_BUF_SIZE=$3
fi

echo $CKPT_FILE
xterm -e "python thread_inf.py --camera=$PORT -f=$F_BUF_SIZE" &
sleep 20 
python popnn4inference.py --ckpt=$CKPT_FILE
