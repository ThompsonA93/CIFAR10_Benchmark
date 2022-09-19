#!/bin/bash
start=0
end=0

for (( i=$start; i<=$end; i++ ))
do   
    echo "[$i/$end] Script iteration"
    python3 src/01_CIFAR10_SKlearn.py
    echo ""
    echo ""
done
echo "Script finished."
echo ""
