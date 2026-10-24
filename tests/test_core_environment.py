from importlib.metadata import version

import gymnasium
import numpy
import torch


def test_validated_dependency_versions_are_loaded():
    assert torch.__version__.split("+")[0] == "2.14.1"
    assert numpy.__version__ == "2.4.6"
    assert gymnasium.__version__ == "1.4.0"
    assert version("pytest") == "9.1.1"


def test_torch_cpu_tensor_and_numpy_interoperate():
    array = numpy.asarray([1.0, 2.0, 3.0], dtype=numpy.float32)
    tensor = torch.from_numpy(array)
    assert tensor.device.type == "cpu"
    assert tensor.dtype == torch.float32
    assert tensor.square().sum().item() == 14.0
