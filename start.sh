#!/bin/bash -e
# NVIDIA_RUNTIME=none ./start.sh  -> skip nvidia docker runtime/gpus args (e.g. on a storage-only node)
# DEVICE=cpu ./start.sh          -> build on the vLLM CPU image (see doc/usage/vllm-cpu-inference.en.md)
source setvars.sh

docker build -f docker/Dockerfile.dev -t "$IAXL_DEV_DOCKER_IMAGE" . \
    --build-arg "BASE=$IAXL_BASE_DOCKER_IMAGE" \
    --build-arg "DEVICE=$DEVICE" \
    --build-arg "IAXL_QATLIB=${IAXL_QATLIB:-}" \
    --build-arg QATLIB_VERSION \
    --build-arg http_proxy --build-arg https_proxy --build-arg no_proxy

docker run \
    "${DOCKER_RUN_ARGS[@]}" \
    --rm \
    --privileged \
    --pid host \
    --net host \
    -it \
    --name "$CONTAINER_NAME" \
    -v /dev:/dev \
    --entrypoint "" \
    "$IAXL_DEV_DOCKER_IMAGE" \
    bash -c '
        for gid in $(id -G); do
            getent group "$gid" >/dev/null 2>&1 || groupadd -g "$gid" "hostgrp$gid" 2>/dev/null || true
        done
        git config --global --add safe.directory "$PWD"
        pip install -e . --verbose --no-build-isolation
        exec bash'
