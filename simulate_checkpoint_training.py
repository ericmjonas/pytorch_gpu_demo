"""
1. What happens if there's an error in this script? 

"""

import torch
import pickle
import time
import sys
import platform
import os



ITERATION_DURATION_SEC = 2.0
CHECKPOINT_EVERY = 5

def run_checkpoint_sim(max_iters, checkpoint_filename):

    name = platform.uname().node
    dev_count = torch.cuda.device_count()
    t1_start = time.time()

    log_filename = checkpoint_filename + '.log'
    log_fid = open(log_filename, 'a+')

    log_fid.write(f"starting simulation, max_iters={max_iters} checkpoint_filename={checkpoint_filename}\n")
    log_fid.write(f"name={name} dev_count={dev_count}\n")
    log_fid.flush()
    
    start_init = 0
    if os.path.exists(checkpoint_filename):
        checkpoint_data = pickle.load(open(checkpoint_filename, 'rb'))

        start_init = checkpoint_data['iter']
        log_fid.write(f"loaded checkpoint {start_init}\n")
        log_fid.flush()


    for i in range(start_init+1, max_iters):
        print("iteration", i)
        log_fid.write(f"running iteration {i} time={time.time():3.2f}, execution runtime {time.time() - t1_start:3.1f}s\n")
        log_fid.flush()

        time.sleep(ITERATION_DURATION_SEC)
        
        if i % CHECKPOINT_EVERY == 0:
            pickle.dump({'iter': i}, open(checkpoint_filename, 'wb'))
            log_fid.write(f"checkpointing iteration {i} time={time.time():3.2f}\n")
            log_fid.flush()

        
    log_fid.write("training done time={time.time():3.2f}\n")
    log_fid.flush()
        

if __name__ == "__main__":
    max_iters = int(sys.argv[1] )
    checkpoint_filename = sys.argv[2]
    
    
    run_checkpoint_sim(max_iters, checkpoint_filename)
