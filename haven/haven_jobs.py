import os
import time
import sys
import subprocess

from . import haven_utils as hu


def run_exp_list_jobs(exp_list,
                      savedir_base,
                      workdir,
                      run_command,
                      job_config=None,
                      force_run=False,
                      wait_seconds=3):
    """Run the experiments in the cluster.

    Parameters
    ----------
    exp_list : list
        list of experiment dictionaries
    savedir_base : str
        the directory where the experiments are saved
    workdir : str
        main directory of the code
    run_command : str
        the command to be ran in the cluster
    job_config : dict
        dictionary describing the job specifications

    Example
    -------
    Add the following code to the main file.

    >>> elif args.run_jobs:
    >>>    from haven import haven_jobs as hjb
    >>>    job_config = {'volume': ['/mnt:/mnt'],
    >>>                'image': <image>,
    >>>            'bid': '1',
    >>>            'restartable': '1',
    >>>            'gpu': '1',
    >>>            'mem': '20',
    >>>            'cpu': '2',
    >>>            'username':<username>}
    >>>    run_command = ('python trainval.py -ei <exp_id> -sb %s' %  (args.savedir_base))
    >>>    hjb.run_exp_list_jobs(exp_list, 
    >>>                        savedir_base=args.savedir_base, 
    >>>                        workdir=os.path.dirname(os.path.realpath(__file__)),
    >>>                        run_command=run_command,
    >>>                        job_config=job_config)
    """
    add_job_utils()
    import haven_jobs_utils as hju

    # ensure no duplicates in exp_list
    hash_list = set()
    for exp_dict in exp_list:
        exp_id = hu.hash_dict(exp_dict)
        if exp_id in hash_list:
            raise ValueError('duplicate experiments detected...')
        else:
            hash_list.add(exp_id)

    # let the user choose one of these options
    print('%d experiments.' % len(exp_list))
    prompt = ("Type one of the following:\n"
              "  1)'reset' to reset the experiments; or\n"
              "  2)'run' to run the remaining experiments and retry the failed ones; or\n"
              "  3)'status' to view the job status.\n"
              "  4)'logs' to view the job logs.\n"
              "  5)'kill' to kill the jobs.\n"
              "Command: "
              )
    if not force_run:
        command = input(prompt)
    else:
        command = 'run'
    command_list = ['reset', 'run', 'status', 'logs', 'kill']
    if command not in command_list:
        raise ValueError(
            'Command has to be one of these choices %s' % command_list)

    # specify reset flag
    reset = False
    if command == 'reset':
        reset = True

    # specify kill flag
    kill_flag = False
    if command == 'kill':
        kill_flag = True

    if command == 'status':
        # view experiments
        hju.get_job_stats(exp_list, savedir_base=savedir_base,
                          username=job_config['username'])

    elif command == 'logs':
        # view experiments
        hju.get_job_logs(exp_list, savedir_base=savedir_base,
                         username=job_config['username'])
        hju.get_job_errors(exp_list, savedir_base=savedir_base,
                           username=job_config['username'])
        hju.get_job_stats(exp_list, savedir_base=savedir_base,
                          username=job_config['username'])

    else:
        # run experiments
        assert(job_config is not None)
        hju.run_experiments(exp_list, savedir_base, reset=reset,
                            job_config=job_config,
                            workdir=workdir,
                            run_command=run_command,
                            kill_flag=kill_flag)

        # view
        print("checking job status in %d seconds..." % wait_seconds)
        time.sleep(wait_seconds)
        hju.get_job_stats(exp_list, savedir_base=savedir_base,
                          username=job_config['username'])




def add_job_utils():
    """adds the ElementAI plugin for running jobs

    Parameters
    ----------
    savedir_base : str
        [description]
    """
    path = '/mnt/datasets/public/issam/haven_utils'
    if path in sys.path:
        return
    sys.path.append(path)
