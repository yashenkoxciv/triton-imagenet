import os
import json
import gdown
import shutil
import docker
import argparse


def get_save_path(model_name):
    return os.path.join('models', model_name, '1', 'model.pt')


def check_models(model_args, models):
    unknown_models = set(model_args).difference(set(models))
    if unknown_models:
        raise RuntimeError('unknown models ' + ', '.join(unknown_models))


def download_model(model_name, model_files):
    model_gd_file_id = model_files[model_name]
    save_path = get_save_path(model_name)
    if not os.path.exists(save_path):
        gdown.download(id=model_gd_file_id, output=save_path, quiet=False)


def prepare_models(models, model_files, repo_root='repo'):
    for model in models:
        download_model(model, model_files)
    if os.path.exists(repo_root):
        shutil.rmtree(repo_root)
    os.makedirs(repo_root, exist_ok=True)

    for model in models:
        src = os.path.join('models', model)
        dst = os.path.join(repo_root, model)
        shutil.copytree(src, dst)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configures triton repository')
    parser.add_argument('image_version', type=str, help='version of the triton image')
    parser.add_argument('models', nargs='+')
    args = parser.parse_args()

    # load maping from model name to model file (id)
    with open('model_idxs.json', 'r') as f:
        model_files = json.loads(f.read())

    check_models(args.models, model_files.keys())

    prepare_models(args.models, model_files)

    client = docker.from_env()
    client.images.build(
        path='.',
        tag=f'triton-imagenet:{args.image_version}',
        quiet=False
    )
