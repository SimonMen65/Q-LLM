# utils/profiler.py
import os

log_path = None
log_file = None

def init_profiler(output_dir, filename="profile.log"):
    global log_path, log_file
    log_path = os.path.join(output_dir, filename)
    log_file = open(log_path, "w")

def log(s):
    global log_file
    if log_file is not None:
        log_file.write(s + "\n")
        log_file.flush()
