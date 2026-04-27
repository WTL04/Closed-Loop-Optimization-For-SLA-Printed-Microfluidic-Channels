# Use Mambaforge for robust binary management
FROM condaforge/mambaforge:latest

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies required by CadQuery (OCP kernel needs OpenGL)
RUN apt-get update && apt-get install -y \
    wget \
    gnupg \
    software-properties-common \
    libgl1-mesa-glx \
    libglu1-mesa \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# 1. Create the Mamba environment with Python 3.11 and CadQuery
RUN mamba create -n cfd_env python=3.11 cadquery -c conda-forge -y

# 2. Copy and install pip dependencies into the conda env
COPY req.txt /tmp/req.txt
RUN mamba run -n cfd_env pip install --no-cache-dir -r /tmp/req.txt

# 3. Install OpenFOAM v1912 (matches colleague's version)
RUN wget -q -O - https://dl.openfoam.com/add-debian-repo.sh | bash && \
    apt-get update && \
    apt-get install -y openfoam1912-default && \
    rm -rf /var/lib/apt/lists/*

# Activate OpenFOAM environment for all subsequent RUN steps and runtime
# Source it in bashrc so it is available when bash -c is used in docker run
RUN echo "source /usr/lib/openfoam/openfoam1912/etc/bashrc" >> /etc/bash.bashrc

# Set conda env on PATH
ENV PATH="/opt/conda/envs/cfd_env/bin:$PATH"

WORKDIR /case

# Entry point runs through the conda env
ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "cfd_env", "bash", "--rcfile", "/etc/bash.bashrc", "-c"]
CMD ["bash"]
