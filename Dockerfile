FROM ubuntu:16.04

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      python-dev \
      build-essential \
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
      gym \
      pyglet \
      tensorflow && \
    conda install \
      bcolz \
      h5py && \
    conda clean -yt && \
    pip install \
      git+git://github.com/keras-rl/keras-rl.git \
      git+git://github.com/stanfordnmbl/osim-rl.git
          

ENV PYTHONPATH='/src/:$PYTHONPATH'

# ADD . /src

WORKDIR /src

RUN git clone git://github.com/mihrab/rl-exploration.git .

CMD python -m ucb.examples.ddpg_l2run > /src/stdout.log
