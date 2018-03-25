FROM tensorflow/tensorflow:1.2.0-gpu-py3

RUN apt-get update && apt-get install -y libffi-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py && \
	rm get-pip.py

#: package list lives in requirements.txt file.
COPY docker_files/requirements.txt /
RUN pip3 --no-cache-dir install -r /requirements.txt

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH




WORKDIR "/root"
# build ocr project in fs
RUN mkdir training_data
RUN mkdir model_data

RUN mkdir /root/.keras/
RUN mkdir /root/.keras/models

COPY nrc_ocr/model model
COPY nrc_ocr/src src
COPY nrc_ocr/training training

# copy keras config
COPY docker_files/keras.json /root/.keras/
COPY docker_files/models /root/.keras/models/
# Set up notebook config
#COPY docker_files/jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly: https://github.com/ipython/ipython/issues/7062
#COPY docker_files/run_jupyter.sh /root/
# make training data
COPY docker_files/gen_data.sh gen_data.sh
# Expose Ports for TensorBoard (6006), Ipython (8888)
EXPOSE 6006

#WORKDIR "/root"
CMD ipython3 training/training.py
