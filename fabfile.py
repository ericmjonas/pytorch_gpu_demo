from fabric.api import local, env, run, put, cd, task, lcd, path, sudo, get
from fabric.contrib import project
import pickle
import os
import time
import boto3
import socket
from glob import glob
from tqdm import tqdm 
import fnmatch

env.roledefs['cs'] = ['ericj@fe01.ai.cs.uchicago.edu']

env.forward_agent = True

@task
def deploy(): 
    local('git ls-tree --full-tree --name-only -r HEAD > .git-files-list')

    tgt_dir = "/home/ericj/pytorch_gpu_demo2"
    
    project.rsync_project(tgt_dir, local_dir="./",
                          exclude=['*.npy', "*.ipynb", 'data'],
                          extra_opts='--files-from=.git-files-list')
    
    # if 'c65' in env.host:

    #     project.rsync_project("/data/jonas/nmr/s2s/notebooks",
    #                           local_dir=".",
    #                           extra_opts="--include '*.png' --include '*.pdf' --include '*.ipynb'  --include='*/' --exclude='*' " ,
    
    #     upload=False)
    



