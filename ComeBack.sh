#!/usr/bin/bash

Lpwd=$(pwd)

rsync -avz francaem@siam-linux16:/local/francaem/SimulationResult/ $Lpwd

rsync -avz francaem@siam-linux16:/local/francaem/nohup.out $Lpwd

rsync -avz francaem@siam-linux16:/local/francaem/LinearNet.py $Lpwd
