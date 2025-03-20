#!/bin/bash

# Add connection to the X server to enable display inside the Docker container.
# This command allows the root user (in the container) to connect to the local X server.
xhost +si:localuser:root

# Run the Docker container with:
#   --gpus all           -> Give the container access to all available NVIDIA GPUs
#   --privileged         -> Required for full device access (e.g., for ZED camera)
#   -e DISPLAY           -> Pass the host DISPLAY variable for GUI applications
#   -v /tmp/.X11-unix    -> Mount the X11 socket directory so GUIs can be displayed
#   -v /usr/local/zed/resources -> Bind-mount ZED resources to persist AI models between runs
#   guulkittil/jp_zed:36.4.3_4.2 -> The Docker image to run
docker run --gpus all -it --privileged \
    -e DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /usr/local/zed/resources:/usr/local/zed/resources \
    guulkittil/jp_zed:36.4.3_4.2

# Tips for container lifecycle management:
# 1. To exit the container, press "Ctrl + C" or type "exit" and press Enter.
# 2. If the container is still running and you detach (Ctrl + P, Ctrl + Q), you can reattach with:
#      docker attach <container-id>
# 3. To see all containers (including stopped ones):
#      docker ps -a
# 4. To stop a running container gracefully:
#      docker stop <container-id>
# 5. To remove (delete) a container once it is stopped:
#      docker rm <container-id>
#    - Or, add the '--rm' flag to 'docker run' to remove it automatically upon exit.

