IMAGE_NAME=gpu_nf
docker build -t $IMAGE_NAME -f DockerfileGPU .
docker run --gpus all -it -v "$(pwd)":/app/host --network=host -p 8888:8888 $IMAGE_NAME