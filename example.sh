# data_set: ml1m, yelp, amazon

# model(victim model): lgn, mf, ncf

# attacker_list(base attacker): AUSHplus, AUSH, AIA, WGAN, SegmentAttacker, AverageAttacker, BandwagonAttacker, RandomAttacker

# way: 1(baseline), 2(Target), 3(UBA)

# oneitem: Target item

# allseed: random seed

python -u main.py --data_set ml1m --model lgn --attacker_list  AUSHplus --attack_num 300 --oneitem 3116  --allseed 2023 --way 1

