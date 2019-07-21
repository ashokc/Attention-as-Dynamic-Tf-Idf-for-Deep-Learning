#!/bin/bash

for attention in yes no; do
	for objective in colors animals; do
		for mask in yes no; do
			out="attention-$attention-objective-$objective-mask-$mask.out"
			echo "PYTHONHASHSEED=0 ; pipenv run python ./dense_attention.py $attention $objective $mask > $out"
			PYTHONHASHSEED=0 ; pipenv run python ./dense_attention.py $attention $objective $mask > $out
		done
	done
done

