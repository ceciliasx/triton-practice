FROM nvcr.io/nvidia/tritonserver:26.10-py3
ENV PIP_BREAK_SYSTEM_PACKAGES=1
RUN python3 -m pip install --no-cache-dir torch==2.15.0 "torchvision==0.29.2+cu132" --index-url https://download.pytorch.org/whl/cu132 && \
    python3 -m pip install --no-cache-dir nvidia-physicsnemo==2.2.2 && \
    find /usr/local/lib/python3.12/dist-packages -maxdepth 1 -name "*.dist-info" ! -exec test -e "{}/METADATA" \; -exec rm -rf {} +