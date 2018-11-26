FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    python3 \
    python3-dev \
    libfreetype6-dev \
    libpng-dev \
    libzmq3-dev \
    libjpeg-dev \
    libtiff-dev \
    zlib1g-dev \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev 

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# --- DO NOT EDIT OR DELETE BETWEEN THE LINES --- #
# These lines will be edited automatically by parameterized_docker_build.sh. #
# COPY _PIP_FILE_ /
# RUN pip --no-cache-dir install /_PIP_FILE_
# RUN rm -f /_PIP_FILE_

# Install TensorFlow GPU version.
RUN pip3 --no-cache-dir install \
    https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp35-cp35m-linux_x86_64.whl
# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

WORKDIR /app

COPY . /app

ENV LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN pip3 install -e /app
RUN pip3 install -r /app/http-wrapper/requirements.txt

ENV FLASK_APP=/app/http-wrapper/app.py

CMD ["python3", "-m", "flask", "run", "--host", "0.0.0.0", "--port", "5000"]
