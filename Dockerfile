FROM nvcr.io/nvidia/tritonserver:21.12-py3

EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

COPY repo /models

CMD ["tritonserver", "--model-repository=/models"]