# Uplift Modeling for Target User Attacks on Recommender Systems

This repository contains our implementations for **UBA**(an <u>U</u>plift-guided <u>B</u>udget <u>A</u>llocation Framework) and various shilling attack methods including Leg-UP, AIA, WGAN, Random Attack, Average Attack, Segment Attack and Bandwagon Attack.



## Enviroment

```shell
conda env create -f environment.yml
conda activate uba
```



## Data

You can download ML-1M, Yelp, Amazon 



## Usage

### *Attack*

1. Attack LightGCN model with baseline attacker Leg-UP

```shell
python -u main.py --data_set ml1m --model lgn --attacker_list  AUSHplus --attack_num 300 --oneitem 3116  --allseed 2023 --way 1
```

2. Attack mf model with UBA attacker AIA

```shell
python -u main.py --data_set ml1m --model mf --attacker_list  AIA --attack_num 300 --oneitem 3116  --allseed 2023 --way 3
```

### *Dynamic Programming Algorith*

```shell
python DPA.py
```



For more example and parameters, please reference the  example.sh .



*Continue updating...*

