import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient("localhost:8000")
inp = httpclient.InferInput("INPUT0", [3], "FP32")
inp.set_data_from_numpy(np.array([1, 2, 3], dtype=np.float32))
result = client.infer("double", [inp])
print(result.as_numpy("OUTPUT0"))