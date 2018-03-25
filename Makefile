CONTAINER_NAME=training
IMAGE_NAME=docker-registry.zoona.io/predictiveanalytics/nrc_ocr/image
PORTS=-p "6006:6006"
VERSION=latest

MODEL_NAME=cnn3x2maxpool
MODEL_PATH=/root/model_data
TRANING_PATH=/root/training_data
VOLUMES=-v /home/thuso/build/nrc_ocr/stage2:/root/training_data -v /home/thuso/build/nrc_ocr/$(MODEL_NAME):/root/model_data/


default: help

help:
	@echo "Training for vanilla ocr for nrcs"
	@echo ""
	@echo "Usage:"
	@echo "    make build-docker        build the docker image that includes the tool"
	@echo "    make run                 runs training on images"
	@echo "    make run-detached        runs training on images in detached mode"
	@echo "    make stop-detached:      stops training from detached mode"
	@echo "    make training_imgs:      creates training images for training"
	@echo "    make tail-logs:          Looks as StOut records"
	@echo "    make push-docker:        Push image to repo"
	@echo "    make run-bash:           Starts interactive bash mode for docker"
	@echo ""
	@echo "Authors:"
	@echo "    Thuso Simon <thuso@ilovezoona.com>"


build-docker:
    @echo $(IMAGE_NAME):$(VERSION)
	docker build --rm -t $(IMAGE_NAME):$(VERSION) ./

training_imgs:
	docker run -e MODEL_NAME=$(MODEL_NAME) -e MODEL_PATH=$(MODEL_PATH) -e TRANING_PATH=$(TRANING_PATH) \
	--rm $(VOLUMES) --name $(CONTAINER_NAME)
run:
	optirun nvidia-docker run -e MODEL_NAME=$(MODEL_NAME) -e MODEL_PATH=$(MODEL_PATH) -e TRANING_PATH=$(TRANING_PATH) \
	$(VOLUMES) -it $(IMAGE_NAME):$(VERSION)

run-detached:
	optirun nvidia-docker run -e MODEL_NAME=$(MODEL_NAME) -e MODEL_PATH=$(MODEL_PATH) -e TRANING_PATH=$(TRANING_PATH) -d \
	$(VOLUMES) -it $(IMAGE_NAME):$(VERSION)

stop-detached:
	nvidia-docker stop $(IMAGE_NAME)

run-tensorborad:
	docker run $(VOLUMES) -p "6006:6006" -d -it $(IMAGE_NAME):$(VERSION) tensorboard --logdir=model_data/logs

tail-logs:
	nvidia-docker logs -f $(IMAGE_NAME)

test:
	nvidia-docker run --rm -it $(IMAGE_NAME):$(VERSION) nvidia-smi

push-docker:
	docker push $(IMAGE_NAME):$(VERSION)

run-bash:
	@echo running docker under bash [TESTING PURPOSES ONLY]
	optirun nvidia-docker run -e MODEL_NAME=$(MODEL_NAME) -e MODEL_PATH=$(MODEL_PATH) -e TRANING_PATH=$(TRANING_PATH) \
	$(VOLUMES) -it $(IMAGE_NAME):$(VERSION) bash