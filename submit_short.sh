
sbatch -p 2080ti-short --gres=gpu:8 --mem 32000 run_celebA128.sh
sbatch -p 2080ti-short --gres=gpu:8 --mem 32000 run_celebA256.sh
sbatch -p 2080ti-short --gres=gpu:8 --mem 32000 run_celebA512.sh


sbatch -p 1080ti-short --gres=gpu:8 --mem 32000 run_cifar256.sh
sbatch -p 1080ti-short --gres=gpu:8 --mem 32000 run_cifar512.sh