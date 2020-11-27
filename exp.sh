#! /bin/bash

touch test.txt
> test.txt

# Run NoHVD
python3 fashion_mnist.py
sleep 5

# Run HVD x2
horovodrun -np 2 -H localhost:2 python3 fashion_mnist_horovod.py
sleep 5

# Run HVD x4
horovodrun -np 4 -H localhost:4 python3 fashion_mnist_horovod.py
sleep 5

# Run HVD x8
horovodrun -np 8 -H localhost:8 python3 fashion_mnist_horovod.py
sleep 5
