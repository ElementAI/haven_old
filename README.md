### Install

install by `pip install --upgrade git+https://github.com/ElementAI/haven`

install in jupyter by `!{sys.executable} -m pip install git+https://github.com/ElementAI/haven --user`

### Code template for main.py 

```python
from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc


def trainval(exp_dict, savedir_base, reset=False):
    # bookkeeping
    # ---------------

    # get experiment directory
    exp_id = hu.hash_dict(exp_dict)
    savedir = os.path.join(savedir_base, exp_id)

    if reset:
        # delete and backup experiment
        hc.delete_experiment(savedir, backup_flag=True)
    
    # create folder and save the experiment dictionary
    os.makedirs(savedir, exist_ok=True)
    hu.save_json(os.path.join(savedir, "exp_dict.json"), exp_dict)
    print(exp_dict)
    print("Experiment saved in %s" % savedir)

    # Dataset
    # -----------

    # train and val loader
    train_loader = ...
    val_loader = ...
   
    # Model
    # -----------
    model = ...

    # Checkpoint
    # -----------
    model_path = os.path.join(savedir, "model.pth")
    score_list_path = os.path.join(savedir, "score_list.pkl")

    if os.path.exists(score_list_path):
        # resume experiment
        model.set_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = score_list[-1]['epoch'] + 1
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # Train & Val
    # ------------
    print("Starting experiment at epoch %d" % (s_epoch))

    for e in range(s_epoch, exp_dict['max_epoch']):
        score_dict = {}

        # Train the model
        score_dict.update(model.train_on_loader(train_loader))

        # Validate the model
        score_dict.update(model.val_on_loader(val_loader, savedir=os.path.join(savedir_base, exp_dict['dataset']['name'])))
        score_dict["epoch"] = e

        # Visualize the model
        model.vis_on_loader(vis_loader, savedir=savedir+"/images/")

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail()
        hu.torch_save(model_path, model.get_state_dict())
        hu.save_pkl(score_list_path, score_list)
        print("Checkpoint Saved: %s" % savedir)

    print('experiment completed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-d', '--datadir_base', required=True)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-v", "--view_experiments", default=None)
    parser.add_argument("-j", "--run_jobs", default=None)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, "exp_dict.json"))        
        
        exp_list = [exp_dict]
        
    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]


    # Run experiments or View them
    # ----------------------------
    if args.view_experiments:
        # view experiments
        hr.view_experiments(exp_list, savedir_base=args.savedir_base)

    elif args.run_jobs:
        # launch jobs
        from haven import haven_jobs as hj
        hj.run_exp_list_jobs(exp_list, 
                       savedir_base=args.savedir_base, 
                       workdir=os.path.dirname(os.path.realpath(__file__)),
                       username='issam',
                       run_command='python trainval.py -ei <exp_id> -sb %s -d %s' % (args.savedir_base, args.datadir_base),
                       job_utils_path='/mnt/datasets/public/issam/haven_borgy/haven_jobs_utils.py',
                       image='images.borgy.elementai.net/issam.laradji/main',
                       bid=1,
                       mem=20,
                       cpu=2,
                       gpu=1)

    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    datadir_base=args.datadir_base,
                    reset=args.reset)
```


### 1. Train and validate experiments

| Command | Description |
| --- | --- |
| `python trainval.py -e MNIST -sb <savedir_base>`| run mnist experiments and save them at <savedir_base> |
| `python trainval.py -e MNIST -sb <savedir_base> -r 1` | reset the mnist experiments |

### 2. View experiments in command-line

| Command | Description |
| --- | --- |
| `python trainval.py -e MNIST -sb <savedir_base> -v 1` | view the mnist experiments |

### 3. View experiments in jupyter
| Command | Description |
| --- | --- |
| `python trainval.py -e MNIST -sb <savedir_base> -j 1` | create the jupyter file 'view_results.ipynb'|


## Haven structure

| File | Description |
| --- | --- |
| `haven_utils.py` | Global utility functions |
| `haven_results.py` | Functions related to viewing  and manipulating results |
