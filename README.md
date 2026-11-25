# Triton & TensorRT
hands-on practice: serving models with Triton Inference Server, then optimising them with TensorRT.

## run
```cmd
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "%cd%\<step>\model_repository:/models" nvcr.io/nvidia/tritonserver:26.10-py3 tritonserver --model-repository=/models
```
client: `python -m venv .venv` → `.venv\Scripts\pip install "tritonclient[http]" numpy`

## contents
| step | folder | what |
|---|---|---|
| 0 | `step0-server` | Triton server with an empty model repository |
| 1 | `step1-hello` | dummy model (y = 2x) with the Python backend |