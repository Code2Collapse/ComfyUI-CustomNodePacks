# C2C Farm backend image — a FULL ComfyUI instance for the render-farm "Brawn".
# Built by .github/workflows/build_docker.yml and deployed by the infra team
# (Embee) onto AKS node pools / RunPod templates. Kubernetes manifests are
# explicitly out of scope here.
#
# Build args let CI pin exact versions:
#   docker build -f renderfarm/docker/Dockerfile \
#     --build-arg COMFYUI_REF=v0.3.40 -t comfy-c2c-farm-backend .

ARG CUDA_TAG=12.4.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

ARG COMFYUI_REPO=https://github.com/comfyanonymous/ComfyUI.git
ARG COMFYUI_REF=master
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python

WORKDIR /opt
RUN git clone ${COMFYUI_REPO} ComfyUI \
    && cd ComfyUI && git checkout ${COMFYUI_REF}

RUN python -m pip install --upgrade pip \
    && python -m pip install torch torchvision torchaudio --index-url ${TORCH_INDEX} \
    && python -m pip install -r /opt/ComfyUI/requirements.txt

# Custom nodes: cloned at PINNED commits from the manifest (format:
#   <git-url> <commit-or-ref>   — see custom_nodes_manifest.txt).
COPY renderfarm/docker/custom_nodes_manifest.txt /opt/manifest.txt
RUN set -e; cd /opt/ComfyUI/custom_nodes; \
    grep -vE '^\s*(#|$)' /opt/manifest.txt | while read -r url ref; do \
        name=$(basename "$url" .git); \
        echo ">> $name @ $ref"; \
        git clone "$url" "$name"; \
        git -C "$name" checkout "$ref"; \
        if [ -f "$name/requirements.txt" ]; then \
            python -m pip install -r "$name/requirements.txt" || \
            echo "WARN: $name requirements partially failed"; \
        fi; \
    done

# C2C Farm courier needs URL-tolerant loaders on the backend; websocket previews
# and storage SDKs so cloud-save nodes can push outputs back.
RUN python -m pip install websocket-client boto3 azure-storage-blob google-cloud-storage

EXPOSE 8188
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=5)" || exit 1

WORKDIR /opt/ComfyUI
CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
