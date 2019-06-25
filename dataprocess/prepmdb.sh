#!/usr/bin/env bash
python3 creatlmdb.py build --dataset '/data/MLRE-NG/PKL/train.pkl' --db /data/MLRE-NG/mdb/train.mdb
python3 creatlmdb.py build --dataset '/data/MLRE-NG/PKL/test.pkl' --db /data/MLRE-NG/mdb/test.mdb
python3 creatlmdb.py eval --dataset '/data/MLRE-NG/PKL/pn1.pkl' --db /data/MLRE-NG/mdb/pn1.mdb
python3 creatlmdb.py eval --dataset '/data/MLRE-NG/PKL/pn2.pkl' --db /data/MLRE-NG/mdb/pn2.mdb
python3 creatlmdb.py eval --dataset '/data/MLRE-NG/PKL/pn3.pkl' --db /data/MLRE-NG/mdb/pn3.mdb
python3 creatlmdb.py build --dataset '/data/MLRE-NG/PKL/train_r.pkl' --db /data/MLRE-NG/mdb/train_r.mdb
python3 creatlmdb.py build --dataset '/data/MLRE-NG/PKL/test_r.pkl' --db /data/MLRE-NG/mdb/test_r.mdb
python3 creatlmdb.py eval --dataset '/data/MLRE-NG/PKL/pn1_r.pkl' --db /data/MLRE-NG/mdb/pn1_r.mdb
python3 creatlmdb.py eval --dataset '/data/MLRE-NG/PKL/pn2_r.pkl' --db /data/MLRE-NG/mdb/pn2_r.mdb
python3 creatlmdb.py eval --dataset '/data/MLRE-NG/PKL/pn3_r.pkl' --db /data/MLRE-NG/mdb/pn3_r.mdb
