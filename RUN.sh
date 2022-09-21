#!/bin/bash
start=0
end=2

for (( i=$start; i<=$end; i++ ))
do   
    echo "[$i/$end] Script iteration"
    python3 src/01_CIFAR10_SKlearn.py
    python3 src/02_CIFAR10_Keras.py
    echo ""
    echo ""
done
echo "Script finished."
echo ""
