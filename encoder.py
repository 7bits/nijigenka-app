import PIL.Image
import onnxruntime
import numpy as np


def to_tensor(
    image: PIL.Image.Image,
    image_size=(256, 256),
) -> np.ndarray:
    image = image.resize(image_size)
    x = np.asarray(image) / 255.0
    x = np.transpose(x, (2, 0, 1))
    x = (x - 0.5) / 0.5
    x = np.expand_dims(x, axis=0).astype(np.float32)
    return x


class Encoder(object):
    def __init__(
        self,
        model_path: str = 'encoder.onnx',
    ) -> None:
        self.input_name = 'input_0'
        self.output_name = 'output_0'
        self.session = onnxruntime.InferenceSession(model_path, None)

    def predict(self, image: PIL.Image.Image) -> np.ndarray:
        x = to_tensor(image)
        output = self.session.run([self.output_name], {
            self.input_name: x,
        })[0][0]
        return output
