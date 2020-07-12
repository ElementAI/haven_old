import os
import time
import sys
import subprocess
from . import haven_utils as hu
from . import haven_chk as hc
import os
from textwrap import wrap
import time
import copy
import pandas as pd
import numpy as np
import getpass
import pprint
from . import haven_jupyter as hj
from . import haven_ork as ho



class JobManager:
    """Job manager."""

    def __init__(self,
                 exp_list=None,
                 savedir_base=None,
                 workdir=None,
                 job_config=None,
                 verbose=1,
                 role_id=None,
                 account_id=None):
        """[summary]

        Parameters
        ----------
        exp_list : [type]
            [description]
        savedir_base : [type]
            [description]
        workdir : [type], optional
            [description], by default None
        job_config : [type], optional
            [description], by default None
        verbose : int, optional
            [description], by default 1
        """
        self.exp_list = exp_list
        self.job_config = job_config
        self.workdir = workdir
        self.verbose = verbose
        self.role_id = role_id
        self.savedir_base = savedir_base
        self.account_id = account_id

        # define funcs
        self.api = ho.get_api(token=None)
    
    def get_command_history(self, topk=10):
        job_list = self.get_jobs()

        count = 0
        for j in job_list:
            if hasattr(j, 'command'):
                print(count,':',j.command[2])
            if count > topk:
                break
            count += 1

    def get_jobs(self):
        return ho.get_jobs(self.api, role_id=self.role_id)

    def get_jobs_dict(self, job_id_list):
        return ho.get_jobs_dict(self.api, job_id_list)

    def get_job(self, job_id):
        return ho.get_job(self.api, job_id)

    def kill_job(self, job_id):
        return ho.kill_job(self.api, job_id)

    def submit_job(self, command, savedir):
        return ho.submit_job(self.api, 
                             self.account_id, 
                             command, 
                             self.job_config, 
                             self.workdir, 
                             savedir)

    def launch_menu(self, command=None, exp_list=None, get_logs=False, wait_seconds=3):
        exp_list = exp_list or self.exp_list
        summary_dict = self.get_summary(get_logs=False, exp_list=exp_list)

        print("\nTotal Experiments:", len(exp_list))
        print("Experiment Status:", summary_dict['status'])
        prompt = ("\nMenu:\n"
                  "  0)'ipdb' run ipdb for an interactive session; or\n"
                  "  1)'reset' to reset the experiments; or\n"
                  "  2)'run' to run the remaining experiments and retry the failed ones; or\n"
                  "  3)'status' to view the job status; or\n"
                  "  4)'kill' to kill the jobs.\n"
                  "Type option: "
                  )
        
        option = input(prompt)

        option_list = ['reset', 'run', 'status', 'logs', 'kill']
        if option not in option_list:
            raise ValueError(
                'Prompt input has to be one of these choices %s' % option_list)

        if option == 'ipdb':
            import ipdb; ipdb.set_trace()
            print('Example:\nsummary_dict = self.get_summary(get_logs=True, exp_list=exp_list)')

        elif option == 'status':
            # view experiments
            for state in ['succeeded', 'running', 'queuing', 'failed']:
                n_jobs = len(summary_dict[state])
                if n_jobs:
                    print('\nExperiments %s: %d' % (state, n_jobs))
                    print(summary_dict[state].head())

            print(summary_dict['status'])
            return

        elif option == 'reset':
            self.verbose = False
            self.launch_exp_list(command=command, exp_list=exp_list, reset=1)

        elif option == 'run':
            self.verbose = False
            self.launch_exp_list(command=command, exp_list=exp_list, reset=0)

        elif option == 'kill':
            self.verbose = False
            self.kill_jobs()

        # view
        print("Checking job status in %d seconds" % wait_seconds)
        time.sleep(wait_seconds)
        summary_dict = self.get_summary(exp_list=exp_list)
        # view experiments
        for state in ['succeeded', 'running', 'queuing', 'failed']:
            n_jobs = len(summary_dict[state])
            if n_jobs:
                print('\nExperiments %s: %d' % (state, n_jobs))
                print(summary_dict[state].head())

        print(summary_dict['status'])

    def launch_exp_list(self, command,  exp_list=None, reset=0, in_parallel=True):
        exp_list = exp_list or self.exp_list

        submit_dict = {}

        if in_parallel:
            pr = hu.Parallel()

            for exp_dict in exp_list:
                exp_id = hu.hash_dict(exp_dict)

                com = command.replace('<exp_id>', exp_id)
                pr.add(self.launch_or_ignore_exp_dict, exp_dict, com, reset, submit_dict)

            pr.run()
            pr.close()

        else:
            for exp_dict in exp_list:
                exp_id = hu.hash_dict(exp_dict)

                com = command.replace('<exp_id>', exp_id)
                self.launch_or_ignore_exp_dict(exp_dict, com, reset, submit_dict)

        pprint.pprint(submit_dict)
        print("%d/%d experiments submitted." % (len([s for s in submit_dict.values() if 'SUBMITTED' in s]),
                                                len(submit_dict)))
        return submit_dict

    def kill_jobs(self, exp_list=None):
        exp_list = exp_list or self.exp_list
        hu.check_duplicates(exp_list)

        pr = hu.Parallel()
        submit_dict = {}

        for exp_dict in exp_list:
            exp_id = hu.hash_dict(exp_dict)
            savedir = os.path.join(self.savedir_base, exp_id)
            fname = get_job_fname(savedir)

            if os.path.exists(fname):
                job_id = hu.load_json(fname)['job_id']
                pr.add(self.kill_job, self.api, job_id)
                submit_dict[exp_id] = 'KILLED'
            else:
                submit_dict[exp_id] = 'NoN-Existent'

        pr.run()
        pr.close()
        pprint.pprint(submit_dict)
        print("%d/%d experiments killed." % (len([s for s in submit_dict.values() if 'KILLED' in s]),
                                             len(submit_dict)))
        return submit_dict

    def launch_or_ignore_exp_dict(self, exp_dict, command, reset, savedir_base=None, submit_dict={}):
        """launch or ignore job.

        It checks if the experiment exist and manages the special casses, e.g.,
        new experiment, reset, failed, job is already running, completed
        """
        # Define paths
        savedir = os.path.join(self.savedir_base, hu.hash_dict(exp_dict))
        fname = get_job_fname(savedir)

        if not os.path.exists(fname):
            # Check if the job already exists
            job_dict = self.launch_exp_dict(exp_dict, savedir, command, job=None)
            job_id = job_dict['job_id']
            message = "SUBMITTED: Launching"

        elif reset:
            # Check if the job already exists
            job_id = hu.load_json(fname).get("job_id")
            self.kill_job(self.api, job_id)
            hc.delete_and_backup_experiment(savedir)

            job_dict = self.launch_exp_dict(exp_dict, savedir, command, job=None)
            job_id = job_dict['job_id']
            message = "SUBMITTED: Resetting"

        else:
            job_id = hu.load_json(fname).get("job_id")
            job = self.get_job( job_id)

            if job.alive or job.state == 'SUCCEEDED':
                # If the job is alive, do nothing
                message = 'IGNORED: Job %s' % job.state

            elif job.state in ["FAILED", "CANCELLED"]:
                message = "SUBMITTED: Retrying %s Job" % job.state
                job_dict = self.launch_exp_dict(exp_dict, savedir, command, job=job)
                job_id = job_dict['job_id']
            # This shouldn't happen
            else:
                raise ValueError('wtf')

        submit_dict[job_id] = message

    def launch_exp_dict(self, exp_dict, savedir, command, job=None,
                   use_toolkit=True):
        """Submit a job job and save job dict and exp_dict."""
        # Check for duplicates
        # if job is not None:
            # assert self._assert_no_duplicates(job)

        fname_exp_dict = os.path.join(savedir, "exp_dict.json")
        hu.save_json(fname_exp_dict, exp_dict)
        assert(hu.hash_dict(hu.load_json(fname_exp_dict))
               == hu.hash_dict(exp_dict))

        # Define paths
        workdir_job = os.path.join(savedir, "code")

        # Copy the experiment code into the experiment folder
        hu.copy_code(self.workdir + "/", workdir_job)

        # Run  command
        job_id = self.submit_job(command, workdir_job, savedir_logs=savedir)

        # Verbose
        if self.verbose:
            print("Job_id: %s command: %s" % (job_id, command))

        job_dict = {"job_id": job_id,
                    "command": command}

        hu.save_json(get_job_fname(savedir), job_dict)

        return job_dict

    def get_summary(self, failed_only=False, columns=None, max_lines=10, wrap_size=8,
                    add_prefix=False, get_logs=True, exp_list=None, savedir_base=None):
        """[summary]

        Returns
        -------
        [type]
            [description]
        """
        # get job ids
        savedir_base = savedir_base or self.savedir_base
        exp_list = exp_list or self.exp_list

        job_id_list = []
        for exp_dict in exp_list:
            exp_id = hu.hash_dict(exp_dict)
            savedir = os.path.join(savedir_base, exp_id)
            fname = get_job_fname(savedir)

            if os.path.exists(fname):
                job_id_list += [hu.load_json(fname)["job_id"]]

        jobs_dict = self.get_jobs_dict(job_id_list)

        # fill summary
        summary_dict = {'table': [], 'status': [],
                        'logs_failed': [], 'logs': []}
        for exp_dict in exp_list:
            result_dict = {}
            for k in exp_dict:
                if isinstance(columns, list) and k not in columns:
                    continue
                if add_prefix:
                    k_new = "(hparam) " + k
                else:
                    k_new = k
                result_dict[k_new] = exp_dict[k]

            result_dict = hu.flatten_column(result_dict)
            result_dict['exp_dict'] = exp_dict
            exp_id = hu.hash_dict(exp_dict)
            savedir = os.path.join(savedir_base, exp_id)
            # result_dict["exp_id"] = '\n'.join(wrap(exp_id, wrap_size))
            result_dict["exp_id"] = exp_id
            fname = get_job_fname(savedir)

            # Job results
            result_dict["job_id"] = None
            result_dict["job_state"] = 'NEVER LAUNCHED'

            if os.path.exists(fname):
                job_dict = hu.load_json(fname)
                job_id = job_dict["job_id"]
                if job_id not in jobs_dict:
                    continue

                fname_exp_dict = os.path.join(savedir, "exp_dict.json")
                job = jobs_dict[job_id]
                result_dict['started_at'] = hu.time_to_montreal(fname_exp_dict)
                result_dict["job_id"] = job_id
                result_dict["job_state"] = job.state
                result_dict["restarts"] = len(job.runs)

                summary_dict['table'] += [copy.deepcopy(result_dict)]

                if hasattr(job, 'command'):
                    result_dict["command"] = job.command[2]
                else:
                    result_dict["command"] = None

                if get_logs:
                    if job.state == "FAILED":
                        fname = os.path.join(savedir, "err.txt")
                        if os.path.exists(fname):
                            result_dict["logs"] = hu.read_text(
                                fname)[-max_lines:]
                            summary_dict['logs_failed'] += [result_dict]
                        else:
                            if self.verbose:
                                print('%s: err.txt does not exist' % exp_id)
                    else:
                        fname = os.path.join(savedir, "logs.txt")
                        if os.path.exists(fname):
                            result_dict["logs"] = hu.read_text(
                                fname)[-max_lines:]
                            summary_dict['logs'] += [result_dict]
                        else:
                            if self.verbose:
                                print('%s: logs.txt does not exist' % exp_id)
            else:
                result_dict['job_state'] = 'NEVER LAUNCHED'
                summary_dict['table'] += [copy.deepcopy(result_dict)]
        # get info
        df = pd.DataFrame(summary_dict['table'])

        # if columns:
        #     df = df[[c for c in columns if (c in df.columns and c not in ['err'])]]

        if "job_state" in df:
            stats = np.vstack(
                np.unique(df['job_state'].fillna("NaN"), return_counts=True)).T
            status = ([{a: b} for (a, b) in stats])
        else:
            df['job_state'] = None

        df = hu.sort_df_columns(df)
        summary_dict['status'] = status
        summary_dict['table'] = df
        summary_dict['queuing'] = df[df['job_state'] == 'QUEUING']
        summary_dict['running'] = df[df['job_state'] == 'RUNNING']
        summary_dict['succeeded'] = df[df['job_state'] == 'SUCCEEDED']
        summary_dict['failed'] = df[df['job_state'] == 'FAILED']

        return summary_dict

    def _assert_no_duplicates(self, job_new=None, max_jobs=500):
        # Get the job list
        jobList = self.get_jobs()

        # Check if duplicates already exist in job
        command_dict = {}
        for job in jobList:

            if hasattr(job, 'command'):
                if job.command is None:
                    continue
                job_python_command = job.command[2]
            else:
                job_python_command = None

            if job_python_command is None:
                continue
            elif job_python_command not in command_dict:
                command_dict[job_python_command] = job
            else:
                print("Job state", job.state, "Job command",
                      job_python_command)
                raise ValueError("Job %s is duplicated" % job_python_command)

        # Check if the new job causes duplicate
        if job_new is not None:
            if job_new.command[2] in command_dict:
                job_old_id = command_dict[job_new.command[2]].id
                raise ValueError("Job exists as %s" % job_old_id)

        return True


def get_job_fname(savedir):
    if os.path.exists(os.path.join(savedir, "borgy_dict.json")):
        # for backward compatibility
        fname = os.path.join(savedir, "borgy_dict.json")
    else:
        fname = os.path.join(savedir, 'job_dict.json')

    return fname



def run_exp_list_jobs(exp_list,
                      savedir_base,
                      workdir,
                      run_command,
                      job_config=None,
                      force_run=False,
                      wait_seconds=3,
                      account_id=None,
                      use_toolkit=False,
                      submit_in_parallel=True):
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
    >>>    job_config = {'data': <data>,
    >>>                  'image': <image>,
    >>>                  'bid': '1',
    >>>                  'restartable': '1',
    >>>                  'gpu': '1',
    >>>                  'mem': '20',
    >>>                  'cpu': '2'}
    >>>    run_command = ('python trainval.py -ei <exp_id> -sb %s' %  (args.savedir_base))
    >>>    hjb.run_exp_list_jobs(exp_list, 
    >>>                          savedir_base=args.savedir_base, 
    >>>                          workdir=os.path.dirname(os.path.realpath(__file__)),
    >>>                          run_command=run_command,
    >>>                          job_config=job_config)
    """
    # let the user choose one of these options
    jm = JobManager(exp_list,
                    savedir_base,
                    run_command=run_command,
                    workdir=workdir,
                    job_config=job_config,
                    verbose=1,
                    account_id=account_id,
                    token=token,
                    force_run=force_run,
                    use_toolkit=use_toolkit,
                    submit_in_parallel=submit_in_parallel)

    jm.run()