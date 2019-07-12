#!/usr/bin/env bash
#python3 r.py -gpu 1 train
#python3 r.py -gpu 1 eval
python3 edr-tl.py -gpu 1 train
python3 edr-tl.py -gpu 1 eval
python3 er-tl.py -gpu 1 train
python3 er-tl.py -gpu 1 eval
python3 dr-tl.py -gpu 1 train
python3 dr-tl.py -gpu 1 eval