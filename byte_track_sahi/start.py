#!/usr/bin/python3
import os
from subprocess import Popen, PIPE
from tools.config import Config

def run_process_2(cmd, prefix=None, suffix=None, cwd = None,
                    use_pipe=False, use_shell = False):
    if cwd == None:
        cwd = os.getcwd()
    print("Running process with command: {}".format(cmd),flush=True)
    cmd_pcs = cmd.split()
    p = Popen(cmd_pcs, cwd = cwd, 
            stdout = PIPE if use_pipe else None,
            stderr = PIPE if use_pipe else None,
            shell = use_shell)
    print("Running process with command: {}".format(cmd),flush=True)

    return p

if __name__ == "__main__":
    cfg = Config()
    model = cfg.model_to_run
    command = {'sahi_yolo7':'python ./tools/sahi_track_v7.py',
               'sahi_yoloX':'python ./byte_track_sahi/tools/sahi_track_vX.py',
               'norm_yoloX':'python tools/demo_track.py'}   
    
   
    print(os. system('nvidia-smi'))
    all_process={}
    p_id=0
    
    p = run_process_2(command[model])
    all_process["P{}".format(p_id)] = [p, True]
    p.wait()   


