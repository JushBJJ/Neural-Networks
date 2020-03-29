#!/bin/bash
while true; do
ps -C /home/jush/Neural-Networks/model1.py -o pid=,%mem=,vsz= >> /home/jush/Neural-Networks/mem.log
gnuplot /home/jush/Neural-Networks/show_mem.plt
sleep 1
done
