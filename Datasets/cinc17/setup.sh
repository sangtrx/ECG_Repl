#!/bin/bash
url=https://archive.physionet.org/challenge/2017/

curl -s -O $url/training2017.zip
unzip -q training2017.zip -d .
rm -r training2017.zip
curl -s -O $url/sample2017.zip
unzip -q sample2017.zip -d .
rm -r sample2017.zip
curl -s -O $url/REFERENCE-v3.csv 

python build_cinc17_dataset.py