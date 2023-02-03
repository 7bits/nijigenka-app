---
title: Nijigenka
emoji: 👁
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 3.17.1
app_file: app.py
pinned: false
license: mit
python_version: 3.8.9
---

Photos stylisation using deep neural networks: anime and art style.

Read more at [7bits.it/portfolio/nijigenka-ai](https://7bits.it/portfolio/nijigenka-ai).

## Local development

Install gradio `pip install gradio==3.17.1`.

Install other deps: `pip install -r requirements.txt`.

Run application `python app.py --models_repo_id=/path/to/dir/with/models`. You can ommit `--models_repo_id` if you would like to download models from the hugging face hub.