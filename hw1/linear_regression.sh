#!/bin/bash
python2 parseData.py
python2 train_original.py
python2 parseTest.py
python2 giveansLinear.py
