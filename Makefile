build:
	sudo docker build -t jax-cuda .

run:
	sudo docker run --gpus all -it ruben-cuda python3.8 main.py
