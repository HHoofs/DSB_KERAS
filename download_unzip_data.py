import zipfile
import os
from subprocess import run
import subprocess as sub


def unzip_in_tmp():
    os.mkdir('tmp')
    with zipfile.ZipFile("stage1_train.zip", "r") as zip_ref:
        zip_ref.extractall("tmp")


def remove_tmp():
    os.rmdir('tmp')


if __name__ == '__main__':
    unzip_in_tmp()
