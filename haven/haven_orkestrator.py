from . import haven_utils as hu
from . import haven_chk as hc
import os
import eai_toolkit_client
import time
import copy
import pandas as pd
import numpy as np 
import getpass
import pprint
import requests
from eai_toolkit_client.rest import ApiException


def submit_job(command, job_config, workdir, savedir_logs=None):
    """
    Launches a job in borgy

    Parameters
    ----------
    command: str
        command you wish to submit
    job_config: dict
        dictionary that should contain keys that correspond to borgy cli's arguments
    workdir: str
        path to where the borgy should run the code
    savedir_logs: str
        path to where the logs are exported (if needed)

    Example
    -------
        import os
        import haven_jobs_utils as hju
        
        job_config = {'data': ['/mnt:/mnt'],
                'image': 'images.borgy.elementai.net/issam/main',
                'bid': '5',
                'restartable': '1',
                'gpu': '1',
                'mem': '50',
                'cpu': '2'}
                
        command = 'echo $PATH'
        job_id = hju.submit_job(command=command, 
                       job_config=job_config, 
                       workdir=os.path.dirname(os.path.realpath(__file__)))
    """
    eai_command = get_job_command(job_config, command, savedir_logs, workdir)
    job_id = hu.subprocess_call(eai_command).replace("\n", "")

    return job_id



def get_api(token=None):
    # Get Borgy API
    jobs_url = 'https://console.elementai.com'
    config = eai_toolkit_client.Configuration()
    config.host = jobs_url

    api_client = eai_toolkit_client.ApiClient(config)

    if token is None:
        try:
            token_url = 'https://internal.console.elementai.com/v1/token'
            r = requests.get(token_url)
            r.raise_for_status()
            token = r.text
        except requests.exceptions.HTTPError as errh:
            # Perhaps do something for each error
            raise SystemExit(errh)
        except requests.exceptions.ConnectionError as errc:
            raise SystemExit(errc)
        except requests.exceptions.Timeout as errt:
            raise SystemExit(errt)
        except requests.exceptions.RequestException as err:
            raise SystemExit(err)

    api_client.set_default_header('Authorization', 'Bearer {}'.format(token))

    # create an instance of the API class
    api = eai_toolkit_client.JobApi(api_client)

    return api 
    
def get_jobs_dict(api, job_id_list, query_size=20):
    # get jobs
    "id__in=64c29dc7-b030-4cb0-8c51-031db029b276,52329dc7-b030-4cb0-8c51-031db029b276"

    jobs = []
    for i in range(0, len(job_id_list), query_size):
        job_id_string = "id__in="
        for job_id in  job_id_list[i:i + query_size]:
            job_id_string += "%s," % job_id
        job_id_string = job_id_string[:-1]
        jobs += api.v1_cluster_job_get(q=job_id_string).items

    jobs_dict = {job.id: job for job in jobs}

    return jobs_dict

def get_job(api, job_id):
    """Get job information."""
    try:
        return api.v1_job_get_by_id(job_id)
    except ApiException as e:
        raise ValueError("job id %s not found." % job_id)

def get_jobs(api):
    return api.v1_me_job_get(limit=1000, 
            order='-created',
            q="alive_recently=True").items
           

def get_job_spec(job_config, command, savedir, workdir):
    _job_config = copy.deepcopy(job_config)
    _job_config['workdir'] = workdir
    
    path_log = os.path.join(savedir, "logs.txt")
    path_err = os.path.join(savedir, "err.txt")
    command_with_logs = '"%s 1>%s 2>%s"' % (command, path_log, path_err)

    _job_config['command'] = ['/bin/bash', '-c', command_with_logs]

    _job_config['resources'] = eai_toolkit_client.JobSpecResources(**_job_config['resources'])
    job_spec = eai_toolkit_client.JobSpec(**_job_config)

    # Return the Borgy command in Byte format
    return job_spec

def get_job_command(job_config, command, savedir, workdir):
    """Compose the borgy submit command."""
    eai_command = "eai job submit "

    # Add the Borgy options
    for k, v in job_config.items():
        if k in ['username']:
            continue
        # Handle all the different type of parameters
        if k == "restartable":
            eai_command += " --%s" % k
        elif k in ["gpu", "cpu", "mem"]:
            eai_command += " --%s=%s" % (k, v)
        elif isinstance(v, list):
            for v_tmp in v:
                eai_command += " --%s %s" % (k, v_tmp)
        elif k in ["userid", "email"]:
            continue
        else:
            eai_command += " --%s %s" % (k, v)
    eai_command += " --workdir %s" % workdir

    # Add the python command to run and the logs file
    if savedir:
        path_log = os.path.join(savedir, "logs.txt")
        path_err = os.path.join(savedir, "err.txt")
        command = '"%s 1>%s 2>%s"' % (command, path_log, path_err)

    eai_command += " -- /bin/bash -c %s" % command

    # Return the Borgy command in Byte format
    return r'''body'''.replace("body", eai_command)

def kill_job(api, job_id):
    """Kill a job job until it is dead."""
    job = get_job(api, job_id)

    if not job.alive:
        print('%s is already dead' % job_id)
    else:
        api.v1_job_delete_by_id(job_id)
        print('%s CANCELLING...' % job_id)
        job = get_job(api, job_id)
        while job.state == "CANCELLING":
            time.sleep(2.0)
            job = get_job(api, job_id)

        print('%s now is dead.' % job_id)
