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
| 2 | *(local only)* | real PyTorch model, Python backend, custom image — not published |

## step 2 — custom image (notes only)
- custom image: `FROM nvcr.io/nvidia/tritonserver:<tag>-py3` + `pip install` the model's libraries
- install `torch` **and** `torchvision` together from the CPU index (`download.pytorch.org/whl/cpu`); PyPI's torchvision on Linux is the CUDA build → `operator torchvision::nms does not exist`
- after pip upgrades a package that came with the base image, an old `*.dist-info` folder without `METADATA` can be left behind and break `importlib.metadata` scans → delete such folders in the Dockerfile
- check correctness: same input through Triton vs plain PyTorch → difference only at float rounding level
- latency breakdown: `curl localhost:8000/v2/models/<name>/stats` → `queue`, `compute_input`, `compute_infer`, `compute_output` (ns / count); client time − Triton time = network
- result: compute dominated; queue + I/O prep inside Triton negligible, HTTP a small share