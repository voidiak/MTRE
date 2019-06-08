#!/usr/bin/env bash
python creatlmdb.py build --dataset 'train.pkl' --db ./mdb/train.mdb
python creatlmdb.py build --dataset 'test.pkl' --db ./mdb/test.mdb
python creatlmdb.py eval --dataset 'pn1.pkl' --db ./mdb/pn1.mdb
python creatlmdb.py eval --dataset 'pn2.pkl' --db ./mdb/pn2.mdb
python creatlmdb.py eval --dataset 'pn3.pkl' --db ./mdb/pn3.mdb
