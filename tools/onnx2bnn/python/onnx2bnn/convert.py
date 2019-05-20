import onnx2bnn._onnx2bnn

import onnx


def convert(model, output, level_str="moderate"):
    if type(model) == str:
        model = onnx.load(model)
        model = model.SerializeToString()
    elif type(model) == onnx.ModelProto:
        model = model.SerializeToString()
    elif type(model) == bytes:
        pass
    else:
        raise RuntimeError(
            "Input of function convert can only be str, onnx.ModelProto or bytes")
    _onnx2bnn.convert(model, output, level_str)
