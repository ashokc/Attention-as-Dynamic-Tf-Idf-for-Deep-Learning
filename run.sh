#!/bin/bash

attention=$1
objective=$2
echo "PYTHONHASHSEED=0 ; pipenv run python ./dense-attention.py $attention $objective"
PYTHONHASHSEED=0 ; pipenv run python ./dense-attention.py $attention $objective

