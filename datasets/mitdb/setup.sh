#!/bin/bash

mkdir data && cd data

url=https://physionet.org/physiobank/database/mitdb/
for i in {100..234}
do
    for ext in 'hea' 'dat' 'atr'
    do
        wget -q -nv $url/$i.$ext
    done
done