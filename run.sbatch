#!/bin/bash
# Your next command here
export CUDA_VISIBLE_DEVICES=1
CUDA_VISIBLE_DEVICES=1 nohup python main_afouier_3d.py > ../co2_out_.txt 2> ../co2_error_.txt &
