import argparse
import functools
import pathlib
import os
from typing import Dict, List, Optional, Tuple
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
    parser.add_argument('--models_repo_id', type=str,
                        default='senior-sigan/nijigenka')
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


def load_examples():
    image_dir = pathlib.Path('examples')
    images = sorted(image_dir.glob('*.jpg'))
    return [[path.as_posix(), 'art'] for path in images]


def join_image_h(im1: PIL.Image.Image, im2: PIL.Image.Image) -> PIL.Image.Image:
    im1 = im1.resize(im2.size)
    dst = PIL.Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def predict(
    image: PIL.Image.Image,
    style: str,
    progress = gr.Progress(),
    *,
    face_aligner: FaceAligner,
    encoder: Encoder,
    generator: Dict[str, Generator],
) -> Tuple[List[PIL.Image.Image], Optional[str]]:
    progress(0, desc="Starting...")
    images = face_aligner.align(image)
    if len(images) == 0:
        error_msg = "Cannot find any face in photo"
        # gradio doesn't support empty list for images carusel, so we create dummy img
        return [PIL.Image.new('RGB', (1, 1))], error_msg

    results = []
    for img in progress.tqdm(images):
        x = encoder.predict(img)
        gen_img = generator[style].predict(x)
        result = join_image_h(img, gen_img)
        results.append(result)

    return results, None


def get_model_path(repo_id: str, filename: str):
    maybe_path = os.path.join(repo_id, filename)
    if os.path.exists(maybe_path):
        print('Using local models')
        return os.path.abspath(maybe_path)
    else:
        return hf_hub_download(
            repo_id,
            filename,
        )


def load_models(repo_id: str):
    encoder_path = get_model_path(
        repo_id,
        'encoder.onnx',
    )
    generator_art_path = get_model_path(
        repo_id,
        'face2art.onnx',
    )
    generator_anime_path = get_model_path(
        repo_id,
        'face2kuvshinov2.onnx',
    )
    shape_predictor_path = get_model_path(
        repo_id,
        'shape_predictor_68_face_landmarks.bin',
    )

    face_aligner = FaceAligner(
        image_size=512,
        shape_predictor_path=shape_predictor_path,
    )
    encoder = Encoder(model_path=encoder_path)
    generator_art = Generator(model_path=generator_art_path)
    generator_anime = Generator(model_path=generator_anime_path)

    return face_aligner, encoder, {'art': generator_art, 'anime': generator_anime}


def main():
    args = parse_args()
    gr.close_all()

    face_aligner, encoder, generator = load_models(args.models_repo_id)
    generator_types = list(generator.keys())

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
            gr.Image(
                type='pil',
                label='Real photo with a face',
            ),
            gr.Radio(
                choices=generator_types,
                type='value',
                value=generator_types[0],
                label='Style',
            ),
        ],
        outputs=[
            gr.Gallery(label='Result'),
            gr.Textbox(label='Error'),
        ],
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
