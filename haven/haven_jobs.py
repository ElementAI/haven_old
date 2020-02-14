from . import haven_utils as hu
import os
import time
import sys 


def run_exp_list_jobs(exp_list, 
                savedir_base, 
                workdir, 
                username,
                run_command,
                job_utils_path,
                image,
                bid,
                mem,
                cpu,
                gpu):
    
    sys.path.append(os.path.dirname(job_utils_path))
    import haven_jobs_utils as hju

    # ensure no duplicates in exp_list
    hash_list = set()
    for exp_dict in exp_list:
        exp_id = hu.hash_dict(exp_dict)
        if exp_id in hash_list:
            raise ValueError('duplicate experiments detected...')
        else:
            hash_list.add(exp_id)

    print('%d experiments.' % len(exp_list))
    prompt = ("Type one of the following:\n"
              "  1)'reset' to reset the experiments; or\n"
              "  2)'run' to run the remaining experiments and retry the failed ones; or\n"
              "  3)'status' to view the experiments' status.\n"
              "  4)'kill' to view the experiments' status.\n"
              "Command: "
                )

    command = input(prompt)
    command_list = ['reset', 'run', 'status', 'kill']
    if command not in command_list:
        raise ValueError('Command has to be one of these choices %s' % command_list)
    
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
        view_experiments(exp_list, savedir_base)
    else:
        # define borgy_config
        borgy_config = {'volume': ['/mnt:/mnt'],
                'image': image,
                'bid': '%d' % bid,
                'restartable': '1',
                'gpu': '%d' % gpu,
                'mem': '%d' % mem,
                'cpu': '%d' % cpu,}

        # run experiments
        # print(borgy_config)
        hju.run_experiments(exp_list, savedir_base, reset=reset, 
                        borgy_config=borgy_config, 
                        username=username, 
                        workdir=workdir,
                        run_command=run_command,
                        kill_flag=kill_flag)

        # view
        n_seconds = 3
        print("checking borgy status in %d seconds..." % n_seconds)
        time.sleep(n_seconds)
        view_experiments(exp_list, savedir_base)


def view_experiments(exp_list, savedir_base):
    import haven_jobs_utils as hju
    print("#exps: %d" % len(exp_list))
    df = hju.get_jobs_df(exp_list, savedir_base=savedir_base)
    print(df)

    hju.print_job_stats(exp_list, savedir_base=savedir_base)