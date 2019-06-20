#!/usr/bin/env bash
python3 creatlmdb.py build --dataset '/data/MLRE-NG/PKL/test_r.pkl' --db /data/MLRE-NG/mdb/test_r.mdb
python3 creatlmdb.py eval --dataset '/data/MLRE-NG/PKL/pn1_r.pkl' --db /data/MLRE-NG/mdb/pn1_r.mdb
python3 creatlmdb.py eval --dataset '/data/MLRE-NG/PKL/pn2_r.pkl' --db /data/MLRE-NG/mdb/pn2_r.mdb
python3 creatlmdb.py eval --dataset '/data/MLRE-NG/PKL/pn3_r.pkl' --db /data/MLRE-NG/mdb/pn3_r.mdb
