import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        # runs once when the model loads; nothing to load for this dummy model
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            x = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()
            y = x * 2
            out = pb_utils.Tensor("OUTPUT0", y.astype(np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[out]))
        return responses