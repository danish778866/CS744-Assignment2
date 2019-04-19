source cluster_utils.sh
start_cluster startserver.py single
python -m AlexNet.scripts.train --mode single --batch_size 64
python -m AlexNet.scripts.train --mode single --batch_size 128
python -m AlexNet.scripts.train --mode single --batch_size 256
terminate_cluster

source cluster_utils.sh
start_cluster startserver.py cluster
python -m AlexNet.scripts.train --mode cluster --batch_size 64
python -m AlexNet.scripts.train --mode cluster --batch_size 128
python -m AlexNet.scripts.train --mode cluster --batch_size 256
terminate_cluster

source cluster_utils.sh
start_cluster startserver.py cluster2
python -m AlexNet.scripts.train --mode cluster2 --batch_size 64
python -m AlexNet.scripts.train --mode cluster2 --batch_size 128
python -m AlexNet.scripts.train --mode cluster2 --batch_size 256
terminate_cluster

