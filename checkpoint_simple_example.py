import parsl
import argparse
import os
from config import config

SOURCE_DIR = os.path.dirname(os.path.abspath(__file__))
    

@parsl.bash_app
def simulate_checkpoint(max_iters, checkpoint_filename,
                        stdout=parsl.AUTO_LOGNAME, stderr=parsl.AUTO_LOGNAME):
    """ This bash app calls the torch_mnist.py example python script from the commandline
    with two kwarg options.
    """


    return f"""cd {SOURCE_DIR}
    python3 simulate_checkpoint_training.py {max_iters} {checkpoint_filename}
    """

    
if __name__ == '__main__':

    # Setting walltime to a small number to trigger failure during training
    config.executors[0].provider.walltime = '00:01:00'

    # Set retries so that each application will rerun 3 times before
    # it is deemed failed.
    config.retries = 1000
    
    parsl.load(config)


    f = simulate_checkpoint(100, os.path.join(SOURCE_DIR, 'test.checkpoint'))

    # fixme is there a way to extract status
    print(f.result())
    
