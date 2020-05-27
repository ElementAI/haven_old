import os
import time
import sys
import subprocess
from . import haven_utils as hu
from . import haven_chk as hc
from . import haven_results as hr
import os
from textwrap import wrap
import time
import copy
import pandas as pd
import numpy as np 
import getpass
import pprint
from . import haven_jupyter as hj


def run_exp_list_jobs(exp_list,
                      savedir_base,
                      workdir,
                      run_command,
                      job_config=None,
                      force_run=False,
                      wait_seconds=3,
                      username=None,
                      account_id=None,
                      token=None,
                      toolkit_mode=False):
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
                username=username, 
                verbose=1,
                account_id=account_id,
                token=token,
                force_run=force_run,
                toolkit_mode=toolkit_mode)
                
    jm.run()


class JobManager:
    """Job manager."""
    def __init__(self, 
                 exp_list, 
                 savedir_base, 
                 run_command=None,
                 workdir=None,
                 job_config=None, 
                 username=None, 
                 verbose=1,
                 account_id=None,
                 token=None,
                 force_run=False,
                 toolkit_mode=False):
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
        username : [type], optional
            [description], by default None
        verbose : int, optional
            [description], by default 1
        """
        hu.check_duplicates(exp_list)

        self.exp_list = exp_list
        self.username = username or getpass.getuser()
        self.job_config = job_config
        self.workdir = workdir
        self.verbose = verbose
        self.savedir_base = savedir_base
        self.account_id = account_id or os.getenv('EAI_TOOLKIT_ACCOUNT_ID')
        self.toolkit_mode = toolkit_mode 
        self.exp_list = exp_list
        self.force_run = force_run
        self.run_command = run_command

        # create an instance of the API class
        if self.toolkit_mode is False:
            from . import haven_borgy as hb
            api_funcs = hb
        else:
            from . import haven_orkestrator as ho
            api_funcs = ho

        self.get_api = lambda token, username: api_funcs.get_api(token=token, username=username)
        self.kill_job = lambda api, job_id: api_funcs.kill_job(api, job_id) 
        self.get_jobs = lambda api, username: api_funcs.get_jobs(api, username) 
        self.get_jobs_dict = lambda api, job_id_list: api_funcs.get_jobs_dict(api, job_id_list) 
        self.get_job = lambda api, job_id: api_funcs.get_job(api, job_id)
        self.submit_job  = lambda api, account_id, command, job_config, workdir, savedir_logs: api_funcs.submit_job(api,
                                        account_id, command, job_config, workdir, savedir_logs)
        
        # define funcs
        self.api = self.get_api(token=token, username=self.username)
    
    def run(self, wait_seconds=3):
        assert self.run_command is not None
        summary_dict = self.get_summary(get_logs=False)
        print("\nTotal Experiments:", len(self.exp_list))
        print("Experiment Status:", summary_dict['status'])
        prompt = ("\nMenu:\n"
                "  0)'results' to view the results using ipdb.\n"
                "  1)'reset' to reset the experiments; or\n"
                "  2)'run' to run the remaining experiments and retry the failed ones; or\n"
                "  3)'status' to view the job status; or\n"
                "  4)'kill' to kill the jobs.\n"
                "Command: "
                )
        if not self.force_run:
            command = input(prompt)
        else:
            command = 'run'

        command_list = ['reset', 'run', 'status','results', 'logs', 'kill']
        if command not in command_list:
            raise ValueError(
                'Command has to be one of these choices %s' % command_list)

        if command == 'results':
            # view experiments
            # table = r
            print('The results are stored in `table` variable')
            import ipdb; ipdb.set_trace()

            return

        if command == 'status':
            # view experiments
            for state in ['succeeded', 'running', 'queuing', 'failed']:
                n_jobs = len(summary_dict[state])
                if n_jobs:
                    print('\nExperiments %s: %d' %(state, n_jobs))
                    print(summary_dict[state].head())

            print(summary_dict['status'])
            return

        elif command == 'reset':
            self.verbose = False
            self.submit_jobs(job_command=self.run_command, reset=1)

        elif command == 'run':
            self.verbose = False
            self.submit_jobs(job_command=self.run_command, reset=0)

        elif command == 'kill':
            self.verbose = False
            self.kill_jobs()

        # view
        print("Checking job status in %d seconds" % wait_seconds)
        time.sleep(wait_seconds)
        summary_dict = self.get_summary()
        # view experiments
        for state in ['succeeded', 'running', 'queuing', 'failed']:
            n_jobs = len(summary_dict[state])
            if n_jobs:
                print('\nExperiments %s: %d' %(state, n_jobs))
                print(summary_dict[state].head())

        print(summary_dict['status'])

        # if not self.force_run:
        #     # create jupyter only when user manually runs a command
        #     hj.create_jupyter(os.path.join('results', 'notebook.ipynb'), savedir_base=savedir_base, print_url=True, 
        #                     create_notebook=False)

    def submit_jobs(self, job_command, reset=0):

        pr = hu.Parallel()
        submit_dict = {}

        for exp_dict in self.exp_list:
            exp_id = hu.hash_dict(exp_dict)
            
            command = job_command.replace('<exp_id>', exp_id)
            pr.add(self._submit_job, exp_dict, command, reset, submit_dict)

        pr.run()
        pr.close()
        pprint.pprint(submit_dict)
        print("%d/%d experiments submitted." % (len([ s for s in submit_dict.values() if 'SUBMITTED' in s]),
                                                len(submit_dict)))
        return submit_dict

    def kill_jobs(self):
        hu.check_duplicates(self.exp_list)

        pr = hu.Parallel()
        submit_dict = {}

        for exp_dict in self.exp_list:
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
        print("%d/%d experiments killed." % (len([ s for s in submit_dict.values() if 'KILLED' in s]),
                                                len(submit_dict)))
        return submit_dict

    def _submit_job(self, exp_dict, command, reset, submit_dict={}):
        """Submit one job.

        It checks if the experiment exist and manages the special casses, e.g.,
        new experiment, reset, failed, job is already running, completed
        """
        # Define paths
        savedir = os.path.join(self.savedir_base, hu.hash_dict(exp_dict))
        fname = get_job_fname(savedir)
        
        if not os.path.exists(fname):
            # Check if the job already exists
            job_dict = self.launch_job(exp_dict, savedir, command, job=None)
            job_id = job_dict['job_id']
            message = "SUBMITTED: Launching"

        elif reset:
            # Check if the job already exists
            job_id = hu.load_json(fname).get("job_id")
            self.kill_job(self.api, job_id)
            hc.delete_and_backup_experiment(savedir)

            job_dict = self.launch_job(exp_dict, savedir, command, job=None)
            job_id = job_dict['job_id']
            message = "SUBMITTED: Resetting"

        else:
            job_id = hu.load_json(fname).get("job_id")
            job = self.get_job(self.api, job_id)

            if job.alive or job.state == 'SUCCEEDED':
                # If the job is alive, do nothing
                message = 'IGNORED: Job %s' % job.state
                
            elif job.state in ["FAILED", "CANCELLED"]:
                message = "SUBMITTED: Retrying %s Job" % job.state
                job_dict = self.launch_job(exp_dict, savedir, command, job=job)
                job_id = job_dict['job_id']
            # This shouldn't happen
            else:
                raise ValueError('wtf')
        
        submit_dict[job_id] = message


    def launch_job(self, exp_dict, savedir, command, job=None,
                   toolkit_mode=True):
        """Submit a job job and save job dict and exp_dict."""
        # Check for duplicates
        if job is not None:
            assert self._assert_no_duplicates(job)

        fname_exp_dict = os.path.join(savedir, "exp_dict.json")
        hu.save_json(fname_exp_dict, exp_dict)
        assert(hu.hash_dict(hu.load_json(fname_exp_dict)) == hu.hash_dict(exp_dict))
        
        # Define paths
        workdir_job = os.path.join(savedir, "code")

        # Copy the experiment code into the experiment folder
        hu.copy_code(self.workdir + "/", workdir_job)

        # Run  command
        job_id = self.submit_job(self.api, self.account_id, command, self.job_config, workdir_job, savedir_logs=savedir)

        # Verbose
        if self.verbose:
            print("Job_id: %s command: %s" % (job_id, command))

        job_dict = {"job_id": job_id, 
                    "command":command}

        hu.save_json(get_job_fname(savedir), job_dict)

        return job_dict

    def get_summary(self, failed_only=False, columns=None, max_lines=10, wrap_size=8, 
                     add_prefix=False, get_logs=True):
        """[summary]
        
        Returns
        -------
        [type]
            [description]
        """
        # get job ids
        job_id_list = []
        for exp_dict in self.exp_list:
            exp_id = hu.hash_dict(exp_dict)
            savedir = os.path.join(self.savedir_base, exp_id)
            fname = get_job_fname(savedir)

            if os.path.exists(fname):
                job_id_list += [hu.load_json(fname)["job_id"]]

        jobs_dict = self.get_jobs_dict(self.api, job_id_list)

        # fill summary
        summary_dict = {'table':[], 'status':[], 'logs_failed':[], 'logs':[]}
        for exp_dict in self.exp_list:
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
            savedir = os.path.join(self.savedir_base, exp_id)
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
                            result_dict["logs"] = hu.read_text(fname)[-max_lines:]
                            summary_dict['logs_failed'] += [result_dict]
                        else:
                            if self.verbose:
                                print('%s: err.txt does not exist' % exp_id)
                    else:
                        fname = os.path.join(savedir, "logs.txt")
                        if os.path.exists(fname):
                            result_dict["logs"] = hu.read_text(fname)[-max_lines:]
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
            stats = np.vstack(np.unique(df['job_state'].fillna("NaN"),return_counts=True)).T
            status = ([{a:b} for (a,b) in stats])
        else:
            df['job_state'] = None

        df =  hu.sort_df_columns(df)
        summary_dict['status'] = status
        summary_dict['table'] = df
        summary_dict['queuing'] = df[df['job_state']=='QUEUING'] 
        summary_dict['running'] = df[df['job_state']=='RUNNING'] 
        summary_dict['succeeded'] = df[df['job_state']=='SUCCEEDED'] 
        summary_dict['failed'] = df[df['job_state']=='FAILED']
        
        return summary_dict

    def _assert_no_duplicates(self, job_new=None, max_jobs=500):
        # Get the job list
        jobList = self.get_jobs(self.api, self.username)

        # Check if duplicates already exist in job
        command_dict = {}
        for job in jobList:
            if hasattr(job, 'command'):
                job_python_command = job.command[2]
            else:
                job_python_command = None
            if job_python_command not in command_dict:
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
    