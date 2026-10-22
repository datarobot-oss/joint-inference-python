# Appended to container/Dockerfile in the deploy build context, so DataRobot's
# server-side build bakes the checkpoint into the last stage. The base image
# itself stays free of weights.
FROM runtime AS bundled

COPY checkpoint/jointfm.pt /opt/jointfm-inference/checkpoint/jointfm.pt
ENV JOINTFM_INFERENCE_CHECKPOINT_PATH=/opt/jointfm-inference/checkpoint/jointfm.pt
