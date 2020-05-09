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

A library for defining hyperparameters, launching and managing many experiments, and visualizing their results.

In the example below, we will write a minimal [codebase](https://github.com/ElementAI/haven/tree/master/examples/minimal) to get us started. 

### Getting Started

The following 4 steps helps in setting up the codebase. 

1. [Define the hyperparameters;](#1-define-the-hyperparameters)
2. [Write the codebase;](#2-write-the-codebase)
3. [Run the experiments; and](#3-run-the-experiments)
4. [Visualize the results.](#4-visualize-the-results)

* [Examples Here](#examples)

### Install
```
$ pip install --upgrade git+https://github.com/ElementAI/haven
```

<!-- /home/issam/Research_Ground/haven/ -->

#### 1. Define the Hyperparameters

Create `exp_configs.py` and add the following dictionary. The experiment group `mnist` defines hyperparameters for comparing learning rates against MNIST. 

```python
# Compare between two learning rates for the same model and dataset
EXP_GROUPS = {'mnist':
                [
                 {'lr':1e-3, 'model':'mlp', 'dataset':'mnist', 'max_epoch':10},
                 {'lr':1e-4, 'model':'mlp', 'dataset':'mnist', 'max_epoch':10}
                  ]
}
```


#### 2. Write the Codebase

A minimal codebase can includee 3 files.

- `trainval.py` for the main loop training and validation loop
- `datasets.py` for selecting the dataset based on the hyperparameter
- `models.py` for selecting the model based on the hyperparameter

##### 2.1 Create the training and validation loop

Create `trainval.py` with the code below: 

```python
import os
import argparse
import pandas as pd
import pprint
import torch 

import exp_configs
import models
import datasets

from haven import haven_utils as hu
from haven import haven_chk as hc
from haven import haven_jobs as hj


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
    hu.save_json(os.path.join(savedir, 'exp_dict.json'), exp_dict)
    pprint.pprint(exp_dict)
    print('Experiment saved in %s' % savedir)

    # Dataset
    # -----------

    # train loader
    train_loader = datasets.get_loader(dataset_name=exp_dict['dataset'], datadir=savedir_base, 
                                        split='train')

    # val loader
    val_loader = datasets.get_loader(dataset_name=exp_dict['dataset'], datadir=savedir_base, 
                                     split='val')

    # Model
    # -----------
    model = models.get_model(model_name=exp_dict['model'])

    # Checkpoint
    # -----------
    model_path = os.path.join(savedir, 'model.pth')
    score_list_path = os.path.join(savedir, 'score_list.pkl')

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
    print('Starting experiment at epoch %d' % (s_epoch))

    for e in range(s_epoch, exp_dict['max_epoch']):
        score_dict = {}

        # Train the model
        train_dict = model.train_on_loader(train_loader)

        # Validate the model
        val_dict = model.val_on_loader(val_loader)

        # Get metrics
        score_dict['train_loss'] = train_dict['train_loss']
        score_dict['val_acc'] = val_dict['val_acc']
        score_dict['epoch'] = e

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print(score_df.tail())
        hu.torch_save(model_path, model.get_state_dict())
        hu.save_pkl(score_list_path, score_list)
        print('Checkpoint Saved: %s' % savedir)

    print('experiment completed')
```

##### 2.2 Add the MNIST dataset

Create `datasets.py` with the following code:

```python
import torchvision

from torch.utils.data import DataLoader


def get_loader(dataset_name, datadir, split, batch_size):
    if dataset_name == 'mnist':
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,))])

        if split == 'train':
            train = True 
        else:
            train = False 

        dataset = torchvision.datasets.MNIST(datadir,
                                                train=train,
                                                download=True,
                                                transform=transform)
        loader = DataLoader(dataset, shuffle=True,
                                  batch_size=batch_size)
        
        return loader
```

##### 2.3 Add the MLP model

Create `models.py` with the following code:

```python
import torch
import tqdm

from torch import nn


def get_model(model_name):
    if model_name == 'mlp':
        return MLP()

class MLP(nn.Module):
    def __init__(self, input_size=784, n_classes=10):
        """Constructor."""
        super().__init__()

        self.input_size = input_size
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, 256)])
        self.output_layer = nn.Linear(256, n_classes)

        self.opt = torch.optim.SGD(self.parameters(), lr=1e-3)

    def forward(self, x):
        """Forward pass of one batch."""
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            Z = layer(out)
            out = torch.nn.functional.relu(Z)
        logits = self.output_layer(out)

        return logits
    
    def get_state_dict(self):
        return {'model': self.state_dict(),
                'opt': self.opt.state_dict()} 

    def load_state_dict(self, state_dict):
        self.load_state_dict(state_dict['model'])
        self.opt.load_state_dict(state_dict['opt'])

    def train_on_loader(self, train_loader):
        """Train for one epoch."""
        self.train()
        loss_sum = 0.

        n_batches = len(train_loader)

        pbar = tqdm.tqdm(desc="Training", total=n_batches, leave=False)
        for i, batch in enumerate(train_loader):
            loss_sum += float(self.train_on_batch(batch))

            pbar.set_description("Training - loss: %.4f " % (loss_sum / (i + 1)))
            pbar.update(1)

        pbar.close()
        loss = loss_sum / n_batches

        return {"train_loss": loss}
    
    @torch.no_grad()
    def val_on_loader(self, val_loader):
        """Validate the model."""
        self.eval()
        se = 0.
        n_samples = 0

        n_batches = len(val_loader)
        pbar = tqdm.tqdm(desc="Validating", total=n_batches, leave=False)

        for i, batch in enumerate(val_loader):
            gt_labels = batch[1]
            pred_labels = self.predict_on_batch(batch)

            se += float((pred_labels.cpu() == gt_labels).sum())
            n_samples += gt_labels.shape[0]
            
            pbar.set_description("Validating -  %.4f acc" % (se / n_samples))
            pbar.update(1)

        pbar.close()

        acc = se / n_samples

        return {"val_acc": acc}

    def train_on_batch(self, batch):
        """Train for one batch."""
        images, labels = batch
        images, labels = images, labels

        self.opt.zero_grad()
        probs = torch.nn.functional.log_softmax(self(images), dim=1)
        loss = torch.nn.functional.nll_loss(probs, labels, reduction="mean")
        loss.backward()

        self.opt.step()

        return loss.item()

    def predict_on_batch(self, batch, **options):
        """Predict for one batch."""
        images, labels = batch
        images = images
        probs = torch.nn.functional.log_softmax(self(images), dim=1)

        return probs.argmax(dim=1)
```

#### 3. Run the Experiments

To run the `mnist` experiment group, follow the two steps below.

##### 3.1 Add the 'Main' Script

Add the following script to the bottom of `trainval.py`. This script allows using the command line to run experiment groups like `mnist`.

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
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

    # run experiments
    for exp_dict in exp_list:
        # do trainval
        trainval(exp_dict=exp_dict,
                savedir_base=args.savedir_base,
                datadir_base=args.datadir_base,
                reset=args.reset)
```



##### 3.2 Run trainval.py in Command Line

The following command launches the mnist experiments and saves their results under `../results/`.

```
python trainval.py -e mnist -sb ../results -r 1
```

##### 3.3 Using a job manager

The experiments can be ran in parallel using a job scheduler such as slurm or the orkestrator. The job scheduler can be used with the following script.

```python
# launch jobs
elif args.run_jobs:
        # launch jobs
        from haven import haven_jobs as hjb
        run_command = ('python trainval.py -ei <exp_id> -sb %s -d %s -nw 1' %  (args.savedir_base, args.datadir_base))
        job_config = {'volume': <volume>,
                    'image': <docker image>,
                    'bid': '1',
                    'restartable': '1',
                    'gpu': '4',
                    'mem': '30',
                    'cpu': '2'}
        workdir = os.path.dirname(os.path.realpath(__file__))

        hjb.run_exp_list_jobs(exp_list, 
                            savedir_base=args.savedir_base, 
                            workdir=workdir,
                            run_command=run_command,
                            job_config=job_config)
```

#### 4. Visualize the Results
![](examples/4_results.png)

The following two steps will setup the visualization environment.

##### 1. Install Jupyter

Run the following in command line to install a Jupyter server
```bash
mkdir .jupyter_server
cd .jupyter_server

virtualenv -p python3 .
source bin/activate

pip install jupyter notebook
pip install ipywidgets
pip install --upgrade git+https://github.com/ElementAI/haven

jupyter nbextension enable --py widgetsnbextension --sys-prefix

jupyter notebook --ip 0.0.0.0 --port 9123 \
      --notebook-dir="/home/$USER" --NotebookApp.token="password"
```

##### 2. Create Jupyter

Add following script in a Jupyter cell to launch a dashboard.


```python
from haven import haven_jupyter as hj
from haven import haven_results as hr
from haven import haven_utils as hu

# path to where the experiments got saved
savedir_base = <insert_savedir_base>
exp_list = None

# exp_list = hu.load_py(<exp_config_name>).EXP_GROUPS[<exp_group>]
# get experiments
rm = hr.ResultManager(exp_list=exp_list, 
                      savedir_base=savedir_base, 
                      verbose=0
                     )
# launch dashboard
hj.get_dashboard(rm, vars(), wide_display=True)
```

### Examples

The following folders contain example projects built on this framework.

- [Minimal](https://github.com/ElementAI/haven/tree/master/examples/minimal)
- [Classification](https://github.com/ElementAI/haven/tree/master/examples/classification)
- [Active Learning](https://github.com/ElementAI/haven/tree/master/examples/active_learning)
- [Generative Adversarial Networks](https://github.com/ElementAI/haven/tree/master/examples/gans)
- [Object Counting](https://github.com/ElementAI/haven/tree/master/examples/object_counting)



### Extras

- Create a list of hyperparameters.
- Save a score dictionary at each epoch.
- Launch a score dictionary at each epoch.
- Create and Launch Jupyter.
- View experiment configurations.
- Debug a single experiment
- View scores as a table.
- View scores as a plot.
- View scores as a barchart.


### FAQ

- case: experiment being saved in a different folder than expected.
- reason: the hashing function might have changed between haven versions.

### Contributing

We love contributions!
