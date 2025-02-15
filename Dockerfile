FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/miniconda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN apt-get update && apt-get install -y curl wget && apt-get clean

RUN curl -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy
RUN conda --version
SHELL ["/bin/bash", "-c"]
ADD . /app
WORKDIR /app
RUN /app/_aigcpanel/build-linux-docker-builder.sh


FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/miniconda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN apt-get update && apt-get install -y curl wget && apt-get clean

RUN curl -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy
RUN conda --version
SHELL ["/bin/bash", "-c"]

COPY --from=builder /app /app

WORKDIR /app
