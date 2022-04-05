import argparse
import functools
import pathlib
import gradio as gr
import PIL.Image
from encoder import Encoder

from face_detector import FaceAligner
from generator import Generator
from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


def load_examples():
    image_dir = pathlib.Path('examples')
    images = sorted(image_dir.glob('*.jpg'))
    return [path.as_posix() for path in images]


def predict(
    image: PIL.Image.Image,
    face_aligner: FaceAligner,
    encoder: Encoder,
    generator: Generator,
):
    images = face_aligner.align(image)

    gen_imgs = []
    for img in images:
        x = encoder.predict(img)
        gen_img = generator.predict(x)
        gen_imgs.append(gen_img)

    return gen_imgs


def load_models():
    encoder_path = hf_hub_download(
        'senior-sigan/nijigenka',
        'encoder.onnx',
    )
    generator_path = hf_hub_download(
        'senior-sigan/nijigenka',
        'face2art.onnx',
    )
    shape_predictor_path = hf_hub_download(
        'senior-sigan/nijigenka',
        'shape_predictor_68_face_landmarks.bin',
    )

    face_aligner = FaceAligner(
        image_size=512,
        shape_predictor_path=shape_predictor_path,
    )
    encoder = Encoder(model_path=encoder_path)
    generator = Generator(model_path=generator_path)

    return face_aligner, encoder, generator


def main():
    args = parse_args()
    gr.close_all()

    face_aligner, encoder, generator = load_models()

    func = functools.partial(
        predict,
        face_aligner=face_aligner,
        encoder=encoder,
        generator=generator,
    )
    func = functools.update_wrapper(func, predict)

    iface = gr.Interface(
        fn=func,
        inputs=[
            gr.inputs.Image(type='pil', label='Input')
        ],
        outputs=gr.outputs.Carousel(['image']),
        examples=load_examples(),
        title='Nijigenka: Portrait to Art',
        allow_flagging='never',
        theme='huggingface',
    )
    iface.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
