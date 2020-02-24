<table>
    <thead>
        <tr>
            <th style="text-align:center;"><img src="docs/images/haven_logo.png" width="40%" alt="Image"></th>
        </tr>
    </thead>
    <tbody>
    </tbody>
</table>


- [Documentation](https://haven.readthedocs.io/en/latest/py-modindex.html)  
- Travis Test: ![CircleCI](https://circleci.com/gh/ElementAI/haven.svg)

<!-- ![](docs/images/haven_logo.png){:height="50%" width="50%"} -->
### Install
```
$ pip install --upgrade git+https://github.com/ElementAI/haven
```

### Demo

Perform 4 different experiments on mnist:

```
python example.py -e mnist -sb ../results -r 1
```

Visualize results:

```python
from haven import haven_results as hr

savedir_base='../results'
exp_list = hr.get_exp_list(savedir_base=savedir_base, 
                           filterby_list=[{'dataset':'mnist'])

# get score lists
score_lists = hr.get_score_lists(exp_list, savedir_base=savedir_base)

# get score table
df = hr.get_score_df(exp_list, savedir_base=savedir_base)

# get plot
fig, axis = hr.get_plot(exp_list, savedir_base=savedir_base, x_metric='epoch', y_metric='train_loss', legend_list=['model'])
```

See also example.ipynb to display the results:



### Usage

Create a file `main.py` with the template below: 

```python
import os
import argparse

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

from haven import haven_utils as hu

# Define exp groups for parameter search
EXP_GROUPS = {'mnist':
                hu.cartesian_exp_group({
                    'lr':[1e-3, 1e-4],
                    'batch_size':[32, 64]})
                }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
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
                       workdir=os.path.dirname(os.path.realpath(__file__)))

    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    datadir_base=args.datadir_base,
                    reset=args.reset)
```

## Haven structure

| File | Description |
| --- | --- |
| `haven_utils.py` | Global utility functions |
| `haven_results.py` | Functions related to viewing  and manipulating results |
