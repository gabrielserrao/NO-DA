#!/bin/bash
PID=391868
while kill -0 $PID 2> /dev/null; do
  sleep 1
done

# Your next command here
CUDA_VISIBLE_DEVICES=1 nohup python main_fouier_3d.py > ../co2_out_.txt 2> ../co2_error_.txt &
