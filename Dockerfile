FROM ubuntu:16.04

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      libblas-dev \
      liblapack-dev \
      gfortran \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

# Install conda
ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN wget --quiet --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo "c59b3dd3cad550ac7596e0d599b91e75d88826db132e4146030ef471bb434e9a *Miniconda3-4.2.12-Linux-x86_64.sh" | sha256sum -c - && \
    /bin/bash /Miniconda3-4.2.12-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-4.2.12-Linux-x86_64.sh && \
    echo export PATH=$CONDA_DIR/bin:'$PATH' > /etc/profile.d/conda.sh

# Install Python packages and keras
ENV NB_USER keras
ENV NB_UID 1000

RUN useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    chown $NB_USER $CONDA_DIR -R && \
    mkdir -p /src && \
    chown $NB_USER /src

USER $NB_USER

ARG python_version=3.6

RUN conda install -y python=${python_version} && \
    conda install -c kidzik opensim && \
    pip install --upgrade pip && \
    pip install \
      sklearn_pandas \
      tensorflow && \
    conda install \
      bcolz \
      h5py \
      mkl \
      nose \
      Pillow \
      pandas \
      pyyaml \
      six && \
    pip install keras-rl && \
    pip install git+https://github.com/stanfordnmbl/osim-rl.git &&\
    conda clean -yt

ADD . /src 

ENV PYTHONPATH='/src/:$PYTHONPATH'

WORKDIR /src

CMD python -m ucb.examples.ddpg_l2run
