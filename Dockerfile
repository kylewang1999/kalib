# ! Building base image.
# ----------------------------------------------------------------------------------------------------------
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel AS base

LABEL "project_name"="Kalib"
LABEL version="1.0"
LABEL author=""
LABEL authors="tttang@sjtu.edu.cn"

# Arguments to build Docker Image using CUDA
ARG USE_CUDA=0
ARG TORCH_ARCH=

ENV AM_I_DOCKER=True
ENV BUILD_WITH_CUDA="${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}"
ENV CUDA_VERSION=11.6
ENV CUDA_HOME=/usr/local/${CUDA_VERSION}
# ENV PATH=${CUDA_HOME}/bin:$PATH
# ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
	--mount=type=cache,target=/var/lib/apt,sharing=locked \
	apt update && apt install -y sudo

# https://stackoverflow.com/questions/27701930/how-to-add-users-to-docker-container
ARG USERNAME=appuser
RUN useradd -rm -d /home/${USERNAME} -s /bin/bash -g root -G sudo -u 1001 ${USERNAME}

# Ensure sudo group users are not asked for a password when using sudo command by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

RUN mkdir -p /home/${USERNAME}/Kalib


# https://docs.docker.com/reference/dockerfile/#shell-and-exec-form
RUN sudo rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
	--mount=type=cache,target=/var/lib/apt,sharing=locked \
	apt-get update && apt-get install --no-install-recommends wget ffmpeg=7:* \
	libsm6=2:* libxext6=2:* git=1:* nano=2.* \
	vim=2:* -y \
	&& apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

RUN --mount=type=bind,source=requirements.txt,target=/tmp/requirements.txt \
	--mount=type=cache,target=/root/.cache/pip  \
	cat /tmp/requirements.txt | sed '$d' | xargs -n1 pip install 

WORKDIR /home/${USERNAME}/Kalib
ARG SCRIPTS_DIR=scripts

# TODO: debug purposeing for here.
# COPY . /home/${USERNAME}/Kalib/

# ! Building build image.
# ----------------------------------------------------------------------------------------------------------
FROM base AS build

LABEL description="Including spatial tracker for kpt tracking. Is the minimum dependency required."

COPY . /home/${USERNAME}/Kalib/
WORKDIR /home/${USERNAME}/Kalib

# installing pip dependencies for spatial_tracker
# use mounted pip caches for docker build: https://stackoverflow.com/questions/58018300/using-a-pip-cache-directory-in-docker-builds
RUN --mount=type=cache,target=/root/.cache/pip \
	cd /home/${USERNAME}/Kalib/third_party/spatial_tracker/ \ 
	&& cat requirements.txt | sed '$d' | xargs -n1 pip install \
	&& pip install cupy-cuda11x \
	# && pip install -r requirements.txt \
	&& pip install gdown \
	&& cd /home/${USERNAME}/Kalib 

RUN bash ${WORKDIR}/${SCRIPTS_DIR}/bootstrap_spatial_tracker.sh ${WORKDIR}


COPY ./dataset/test_sample_sim_data/ /home/${USERNAME}/Kalib/dataset/

# Install Kalib as a package.
RUN pip install -e .

# ! Building sam image.
# ----------------------------------------------------------------------------------------------------------
FROM build AS sam_included

LABEL description="Including sam checkpoints. Grounded-Sam can be included to generate foreground mask to potentially increase kpt tracking performance."

RUN bash ${WORKDIR}/${SCRIPTS_DIR}/bootstrap_sam.sh ${WORKDIR}

# ! Building grounded_sam image.
# ----------------------------------------------------------------------------------------------------------
FROM sam_included AS grounded_sam_included

RUN ln -sf ${WORKDIR}/pretrained_checkpoints ${WORKDIR}/third_party/grounded_segment_anything/

# installing pip dependencies for grounded-sam
RUN --mount=type=cache,target=/root/.cache/pip \
	cd /home/${USERNAME}/Kalib/third_party/grounded_segment_anything \ 
	&& pip install -r requirements.txt \
	&& pip install --no-cache-dir wheel \
	&&  pip install --no-cache-dir --no-build-isolation -e GroundingDINO \
	&& cd /home/${USERNAME}/Kalib 


RUN bash ${WORKDIR}/${SCRIPTS_DIR}/bootstrap_grounded_sam.sh ${WORKDIR}


# ! Building droid image.
# ----------------------------------------------------------------------------------------------------------
FROM base AS droid_included

LABEL description="Extracts droid dataset data and export to Kalib-compatible img and json dataset format."


RUN mkdir -p /home/${USERNAME}/Kalib/easycalib && mkdir -p /home/${USERNAME}/Kalib/third_party/droid

COPY ./easycalib/ /home/${USERNAME}/Kalib/easycalib/
COPY ./easycalib_demo.py /home/${USERNAME}/Kalib/
COPY ./third_party/droid /home/${USERNAME}/Kalib/third_party/droid
WORKDIR /home/${USERNAME}/Kalib

RUN --mount=type=cache,target=/root/.cache/pip \
	cd /home/${USERNAME}/Kalib/third_party/droid/ \ 
	&& pip install -e . \
	&& cd /home/${USERNAME}/Kalib 

RUN mkdir -p /home/${USERNAME}/Kalib/dataset/test_extract_droid_data
COPY ./dataset/test_extract_droid_data /home/${USERNAME}/Kalib/dataset/test_extract_droid_data

RUN --mount=type=bind,source=setup.py,target=/home/${USERNAME}/Kalib/setup.py \
	pip install -e .

RUN cd /tmp || exit
ARG ZED_SDK_SAVE_PATH="/tmp/zed_sdk_cu118_ubuntu22_zstd.run"
ADD https://download.stereolabs.com/zedsdk/4.1/cu118/ubuntu22 ${ZED_SDK_SAVE_PATH}

USER ${USERNAME}
WORKDIR /home/${USERNAME}

RUN sudo apt update && sudo apt install -y zstd udev && sudo apt-get clean && sudo apt-get autoremove && sudo rm -rf /var/lib/apt/lists/* \
	&&  sudo chmod +x  ${ZED_SDK_SAVE_PATH} \
	&& sudo bash  ${ZED_SDK_SAVE_PATH} -- silent

USER root
# Fixing dependency issue triggered by droid requirements.txt.
RUN pip uninstall -y opencv-python; pip install opencv-contrib-python==4.10.0.82; pip install "numpy<2.0"; pip install "torchvision"

# CMD python /home/${USERNAME}/Kalib/easycalib/utils/process_droid_data.py --data_dir /home/${USERNAME}/Kalib/dataset/test_extract_droid_data --data_save_path /home/${USERNAME}/Kalib/dataset/processed_droid_data

# ! Building cotracker image.
# ----------------------------------------------------------------------------------------------------------
FROM sam_included AS cotracker_included

LABEL description="CoTrakcer included for dabbling into different kpt tracking module."

RUN cd /home/appuser/Kalib/third_party/cotracker && mkdir -p checkpoints && cd checkpoints

ADD https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth ./cotracker2.pth

RUN --mount=type=bind,source=${SCRIPTS_DIR}/bootstrap_cotracker.sh,target=/tmp/bootstrap_cotracker.sh \
	bash /tmp/bootstrap_cotracker.sh

# ! Building rfu image.
# ----------------------------------------------------------------------------------------------------------
FROM base AS pyrfuniverse_included

LABEL description="Pyrfuniverse testing script for generating the sim-env dataset."

WORKDIR /home/${USERNAME}

RUN mkdir -p /home/${USERNAME}/Kalib/scripts/
COPY ./scripts/ /home/${USERNAME}/Kalib/scripts

RUN mkdir -p /home/${USERNAME}/Kalib/tests/
COPY ./tests/ /home/${USERNAME}/Kalib/tests

WORKDIR /home/${USERNAME}/Kalib

ARG RFU_PORT=5005

EXPOSE ${RFU_PORT}

RUN --mount=type=bind,source=${SCRIPTS_DIR}/bootstrap_pyrfuniverse.sh,target=/tmp/bootstrap_pyrfuniverse.sh \
	bash /tmp/bootstrap_pyrfuniverse.sh ${WORKDIR}