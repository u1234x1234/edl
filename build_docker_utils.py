# -*- coding: utf-8 -*-
import argparse
import logging
import os
import shutil
from typing import List
from subprocess import check_output, run, PIPE

DOCKER_BIN = 'docker'


def _run_command(args: List[str]) -> str:
    out = run(args, stdout=PIPE, encoding='utf-8').stdout.strip()
    return out


def _check_docker_installation():
    version = _run_command([DOCKER_BIN, '--version'])
    logging.debug(f'Docker version: {version}')


def copy_from_image_to_host(image_name: str, src: str, dst: str):
    """Run the docker image `image_name` and copy files from `src` to the host system `dst`
    """
    new_dir = os.path.join(dst, os.path.basename(src))
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
        logging.info(f'Directory {new_dir} has been removed')

    container_id = _run_command([DOCKER_BIN, 'run', '-td', image_name])
    _run_command([DOCKER_BIN, 'cp', f'{container_id}:{src}', f'{dst}'])
    _run_command([DOCKER_BIN, 'stop', container_id])


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    _check_docker_installation()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--name', type=str)
    args = arg_parser.parse_args()

    copy_from_image_to_host('build_openblas', '/work/lib_openblas', './')
    copy_from_image_to_host('build_opencv', '/work/lib_opencv', './')
