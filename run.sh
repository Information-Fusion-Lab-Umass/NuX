IMAGE_NAME=non_gpu_nf
docker build -t $IMAGE_NAME -f Dockerfile .
docker run -it --rm -v "$(pwd)":/app/host -p 8888:8888 $IMAGE_NAME /bin/bash