"""
Questions: 
1. How do we determine how many CPUs/GPUs are used for a given job? it seems if we set nodes-per-worker than the job runs on each? 
"""

import parsl
import argparse
import os

from parsl.config import Config
from parsl.channels import LocalChannel
from parsl.providers import SlurmProvider
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SrunLauncher
from parsl.addresses import address_by_hostname


SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
    

@parsl.bash_app
def simulate_checkpoint(SOURCE_DIR, max_iters, checkpoint_filename,
                        stdout=parsl.AUTO_LOGNAME, stderr=parsl.AUTO_LOGNAME):
    """ This bash app calls the torch_mnist.py example python script from the commandline
    with two kwarg options.
    """
    import os
    

    return f"""cd {SOURCE_DIR}
    python3 simulate_checkpoint_training.py {max_iters} {checkpoint_filename}
    """

# Configure options here:
NODES_PER_JOB = 1
GPUS_PER_NODE = 1
GPUS_PER_WORKER = 1

# Do not modify:
TOTAL_WORKERS = int((NODES_PER_JOB*GPUS_PER_NODE)/GPUS_PER_WORKER)
WORKERS_PER_NODE = int(GPUS_PER_NODE / GPUS_PER_WORKER)
GPU_MAP = ','.join([str(x) for x in range(1,TOTAL_WORKERS + 1)])

config = Config(
    executors=[
        HighThroughputExecutor(
            label="fe.cs.uchicago",
            address=address_by_hostname(),
            max_workers=1,  # Sets #workers per manager.
            provider=SlurmProvider(
                channel=LocalChannel(),
                nodes_per_block=NODES_PER_JOB,
                init_blocks=1,
                partition='geforce',
                scheduler_options=f'#SBATCH --gpus-per-node=rtx2080ti:{GPUS_PER_NODE}',
                # Launch 4 managers per node, each bound to 1 GPU
                # This is a hack. We use hostname ; to terminate the srun command, and start our own
                # DO NOT MODIFY unless you know what you are doing.
                launcher=SrunLauncher(overrides=(f'hostname; srun --ntasks={TOTAL_WORKERS} '
                                                 f'--ntasks-per-node={WORKERS_PER_NODE} '
                                                 f'--gpus-per-task=rtx2080ti:{GPUS_PER_WORKER} '
                                                 f'--gpu-bind=map_gpu:{GPU_MAP}')
                ),
                worker_init='eval "$($WORK/anaconda/bin/conda shell.bash hook)"; conda activate parsl_py3.7', 

                walltime='01:00:00',
            ),
        )
    ],
)


if __name__ == '__main__':

    # Setting walltime to a small number to trigger failure during training
    config.executors[0].provider.walltime = '00:01:30'

    # Set retries so that each application will rerun 3 times before
    # it is deemed failed.
    config.retries = 20
    
    parsl.load(config)


    f = simulate_checkpoint(SOURCE_DIR, 100, os.path.join(SOURCE_DIR, 'test.checkpoint'))

    # fixme is there a way to extract status
    print(f.result())
    
