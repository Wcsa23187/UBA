logname=$10
nohup python -u main.py --data_path=$1 --data_type=$2 --dataset=$3 --model=$4 --epochs=200 --decay=$5 --lr=$6 --layer=$7 --recdim=$8 --dropout=$9 --gpu=$11 > ./log/LightGCN_new2_$2_$3_$4_$5decay_$6lr_$7layer_$8recdim_$9dropout_$logname.txt 2>&1 &

# sh run.sh ../data/ time electronics lgn 1e-4 0.001 3 128 0 log 0