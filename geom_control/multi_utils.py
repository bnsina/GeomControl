from . import control

def runner(namespace):
    job = control.controller(namespace)
    job.run_control()
    