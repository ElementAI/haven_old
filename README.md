<!-- <table>
    <thead>
        <tr>
            <th style="text-align:center;"><img src="docs/images/haven_logo.png" width="40%" alt="Image"></th>
        </tr>
    </thead>
    <tbody>
    </tbody>
</table> -->
# Haven 


A style-guide for creating, managing, and visualizing experiments. 
* [Install](#install)
* [Overview](#overview)
* [Examples](#examples)
* [Features](#features)
* [Contributing](#contributing)

### Install
```
$ pip install --upgrade git+https://github.com/ElementAI/haven
```
### Overview

This guide focuses on readability, reliability, and flexibility. It consists of 4 main steps:

1. [Codebase](#codebase): write code for a machine learning project.
2. [Hyperparameters](#hyperparameters): define a list of experiments.
3. [Experiments](#experiments): run  and manage thousands of experiments.
4. [Visualization](#visualization): aggregate through thousands of results.

### Examples

- [Classification](https://github.com/ElementAI/haven/tree/master/examples/classification)
- [Active Learning](https://github.com/ElementAI/haven/tree/master/examples/active_learning)
- [Generative Adversarial Networks](https://github.com/ElementAI/haven/tree/master/examples/gans)
- [Object Counting](https://github.com/ElementAI/haven/tree/master/examples/object_counting)

### Features

- Create a list of hyperparameters.
- Save a score dictionary at each epoch.
- Launch a score dictionary at each epoch.
- Create and Launch Jupyter.
- View experiment configurations.
- View scores as a table.
- View scores as a plot.
- View scores as a barchart.




#### Codebase

Create a file `main.py` with the template below: 

```python
import os
import argparse

from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
import pandas as pd


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
        print("\n", score_df.tail())
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



#### Hyperparameters
Define an experiment group `mnist` as a list of hyperparameters

```
EXP_GROUPS = {'mnist':
                hu.cartesian_exp_group({
                    'dataset':'mnist',
                    'model':'mlp',
                    'max_epoch':20,
                    'lr':[1e-3, 1e-4],
                    'batch_size':[32, 64]})
                }
```

#### Experiments

Trains a model on mnist across a set of hyperparameters:

```
python example.py -e mnist -sb ../results -r 1
```

#### Visualization

The following two steps will setup the visualization environment.

##### 1. Install Jupyter

```bash
mkdir .jupyter_server
cd .jupyter_server
virtualenv -p python3 .
source bin/activate
pip install jupyter notebook
pip install ipywidgets
jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter notebook --ip 0.0.0.0 --port 9123 \
      --notebook-dir="/home/$USER" --NotebookApp.token=<password>
```

##### 2. Create Jupyter

Shown in example.ipynb.
Add the following two cells to a Jupyter notebook.

##### Cell 1

```python
# Setup variables
# ===============

savedir_base = <savedir_base>
workdir = <workdir>
exp_group_list = ['default']


# exp vars
filterby_list = None
groupby_list = None
verbose = 1

# table vars
columns = None

# plot vars
y_metric='train_loss'
x_metric='epoch'
map_exp_list = []
figsize=(10,5)
title_list=['dataset']
legend_list=['model']
mode='line'

# image vars
image_legend_list = []
n_images=5

# job vars
username = 'anonymous'
columns = None
```


##### Cell 2

```python
# Create vizualizations
# =====================

import pprint 

from ipywidgets import Button, HBox, VBox
from ipywidgets import widgets
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots

from IPython.display import display
from IPython.core.display import Javascript, display

from haven import haven_results as hr
from haven import haven_jupyter as hj

hj.init_datatable_mode()

# Get experiments
rm = hr.ResultManager(workdir=workdir, 
                      exp_group_list=exp_group_list,
                      savedir_base=savedir_base, 
                      filterby_list=filterby_list,
                      groupby_list=groupby_list,
                      verbose=verbose
                     )

tables = widgets.Output()
plots = widgets.Output()
images = widgets.Output()
job_states = widgets.Output()
job_logs = widgets.Output()
job_failed = widgets.Output()

# Display tabs
tab = widgets.Tab(children = [tables, plots, images, job_states, job_logs, job_failed])
tab.set_title(0, 'Tables')
tab.set_title(1, 'Plots')
tab.set_title(2, 'Images')
tab.set_title(3, 'Job States')
tab.set_title(4, 'Job Logs')
tab.set_title(5, 'Job Failed')
display(tab)

# Display tables
with tables:
    exp_table = rm.get_exp_table()
    # Get score table 
    score_table = rm.get_score_table()
    output.clear_output()
    
    display(exp_table)
    display(score_table)

# Display plots
with plots:
    rm.get_plot(y_metric=y_metric, 
            x_metric=x_metric, 
            legend_list=legend_list, 
            map_exp_list=map_exp_list, 
            mode=mode,
            figsize=figsize,
            title_list=title_list)
    show_inline_matplotlib_plots()

# Display images
with images:
    rm.get_images(legend_list=image_legend_list, n_images=n_images)
    show_inline_matplotlib_plots()

# Display job states
# with job_states:
#     table_dict = rm.get_job_summary(username=username)[0]
#     display(table_dict['status'])
#     display(table_dict['table'])

# # Display job failed
# with job_logs:
#     table_dict = rm.get_job_summary(username=username)[0]
 
#     display(table_dict['status'])
#     display(table_dict['table'])
#     for logs in table_dict['logs']:
#          pprint.pprint(logs)
                
# # Display job failed
# with job_failed:
#     table_dict = rm.get_job_summary(username=username)[0]
#     if len(table_dict['failed']) == 0:
#         display('no failed experiments')
#     else:
#         display(table_dict['failed'])
#         for failed in table_dict['logs_failed']:
#              pprint.pprint(failed)
```

To install Haven from a jupyter cell, add the following cell,

```python
import sys
!{sys.executable} -m pip install --upgrade  --no-dependencies 'git+https://github.com/ElementAI/haven' --user
```
<!-- /home/issam/Research_Ground/haven/ -->


### Contributing

We love contributions!