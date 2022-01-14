python3 vocab.py
python3 run.py --train --batch-size=64 --epoches=100 --src-sents-path=/Users/tianwentang/Datasets/chr_en_data/train.chr --tgt-sents-path=/Users/tianwentang/Datasets/chr_en_data/train.en --hidden-size=100 --dropout-rate=0.1 --embed-size=100


# python3 run.py --train --deviece=cuda --batch-size=64 --data-path=/Users/tianwentang/Datasets/translation2019zh/translation2019zh_train.json --valid-iter=100 --embed-size=1000 --dropout-rate=0.1 --hidden-size=100 --epoches=50
# python3 run.py --train --deviece=cuda --batch-size=64 --data-path=E:\Datasets\translation2019zh\translation2019zh_train.json --valid-iter=100 --embed-size=1000 --dropout-rate=0.1 --hidden-size=100 --epoches=50



