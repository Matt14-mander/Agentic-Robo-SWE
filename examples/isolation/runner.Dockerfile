FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends python3 cmake ninja-build g++ clangd \
    && rm -rf /var/lib/apt/lists/*
COPY runner.py /opt/robo/runner.py
