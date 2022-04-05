from xml.etree.ElementTree import PI
import PIL.Image
import onnxruntime
import numpy as np


def to_image(tensor: np.ndarray) -> PIL.Image.Image:
    tensor = tensor * 0.5 + 0.5
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    tensor = np.transpose(tensor, (1, 2, 0))
    return PIL.Image.fromarray(tensor)


class Generator(object):
    def __init__(
        self,
        model_path: str = 'generator.onnx',
    ) -> None:
        self.input_name = 'input_0'
        self.output_name = 'output_0'
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 8
        self.session = onnxruntime.InferenceSession(
            model_path, sess_options=opts)

    def predict(self, x: np.ndarray) -> PIL.Image.Image:
        x = np.expand_dims(x, 0)
        output = self.session.run([self.output_name], {
            self.input_name: x,
        })[0][0]
        return to_image(output)
