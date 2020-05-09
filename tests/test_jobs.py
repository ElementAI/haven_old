import unittest
import numpy as np 
import os, sys
import torch
import shutil, time

path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, path)

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jobs as hjb
from haven import haven_orkestrator as ho
from haven import haven_jupyter as hj


class Test(unittest.TestCase):

    def test_submit_job(self):

        job_config = {'data': ['/mnt:/mnt'],
                'image': 'images.borgy.elementai.net/issam/main',
                'bid': '5',
                'restartable': '1',
                'gpu': '0',
                'mem': '2',
                'cpu': '1'}
        command = 'echo $PATH'
        job_id = ho.submit_job(command=command, 
                       job_config=job_config, 
                       workdir=os.path.dirname(os.path.realpath(__file__)))
        print(job_id)
    
    def test_get_job_stats_logs_errors(self):
        # return
        exp_list = [{'model':{'name':'mlp', 'n_layers':30}, 
                    'dataset':'mnist', 'batch_size':1}]
        savedir_base = '/mnt/datasets/public/issam/tmp'
        job_config = {'data': ['/mnt:/mnt'],
                    'image': 'registry.console.elementai.com/75ce4cee-6829-4274-80e1-77e89559ddfb',
                    'bid': '1',
                    'restartable': '1',
                    'gpu': '1',
                    'mem': '20',
                    'cpu': '2',
                    }
        run_command = ('python example.py -ei <exp_id> -sb %s' %  (savedir_base))
        
        hjb.run_exp_list_jobs(exp_list, 
                            savedir_base=savedir_base, 
                            workdir=os.path.dirname(os.path.realpath(__file__)),
                            run_command=run_command,
                            job_config=job_config,
                            force_run=True,
                            wait_seconds=0)
        assert(os.path.exists(os.path.join(savedir_base, hu.hash_dict(exp_list[0]), 'borgy_dict.json')))
        jm = hjb.JobManager(exp_list=exp_list, savedir_base=savedir_base)
        jm_summary_list = jm.get_summary()
        rm = hr.ResultManager(exp_list=exp_list, savedir_base=savedir_base)
        rm_summary_list = rm.get_job_summary()
        assert(rm_summary_list['table'].equals(jm_summary_list['table']))

        jm.kill_jobs()
        assert('CANCELLED' in jm.get_summary()['status'][0])
        
if __name__ == '__main__':
    unittest.main()

