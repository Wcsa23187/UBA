nohup python -u main.py  --data_path=$1 --data_type=$2 --dataset=$3  --model=$4  --epochs=200  --lr=$5  --dropout=$6  --factor_num=$7  --num_layers=$8  --log_name=$9 --gpu=${10} --GMF_model_path=${11} --MLP_model_path=${12} > ./log/try_mf_$2_$3_$4_$5lr_$6dropout_$7factornum_$8numlayers_$9.txt 2>&1 &

# sh run.sh /storage/shjing/recommendation/causal_discovery/data/amazon_electronics/data_relative_10/split_time time electronics MF 0.001 0 32 5 log 0 None None
