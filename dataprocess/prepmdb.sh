#!/usr/bin/env bash
python3 creatlmdb.py build --dataset '/data/MLRE-NG-archive/PKL/train.pkl' --db /data/MLRE-NG/testmdb/train.mdb
python3 creatlmdb.py build --dataset '/data/MLRE-NG-archive/PKL/test.pkl' --db /data/MLRE-NG/testmdb/test.mdb
python3 creatlmdb.py eval --dataset '/data/MLRE-NG-archive/PKL/pn1.pkl' --db /data/MLRE-NG/testmdb/pn1.mdb
python3 creatlmdb.py eval --dataset '/data/MLRE-NG-archive/PKL/pn2.pkl' --db /data/MLRE-NG/testmdb/pn2.mdb
python3 creatlmdb.py eval --dataset '/data/MLRE-NG-archive/PKL/pn3.pkl' --db /data/MLRE-NG/testmdb/pn3.mdb
python3 creatlmdb.py build --dataset '/data/MLRE-NG-archive/PKL/train_r.pkl' --db /data/MLRE-NG/testmdb/train_r.mdb
python3 creatlmdb.py build --dataset '/data/MLRE-NG-archive/PKL/test_r.pkl' --db /data/MLRE-NG/testmdb/test_r.mdb
python3 creatlmdb.py eval --dataset '/data/MLRE-NG-archive/PKL/pn1_r.pkl' --db /data/MLRE-NG/testmdb/pn1_r.mdb
python3 creatlmdb.py eval --dataset '/data/MLRE-NG-archive/PKL/pn2_r.pkl' --db /data/MLRE-NG/testmdb/pn2_r.mdb
python3 creatlmdb.py eval --dataset '/data/MLRE-NG-archive/PKL/pn3_r.pkl' --db /data/MLRE-NG/testmdb/pn3_r.mdb
