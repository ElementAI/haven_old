import os
import argparse
# import torchvision
import pandas as pd
import pprint

from src import datasets, models

from haven import haven_utils as hu
# from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jupyter as hj


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
    train_loader = datasets.get_loader(
        dataset_name=exp_dict['dataset'], datadir=savedir_base, split='train',
        batch_size=exp_dict['batch_size'])

    # val loader
    val_loader = datasets.get_loader(
        dataset_name=exp_dict['dataset'], datadir=savedir_base, split='val',
        batch_size=exp_dict['batch_size'])

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


# Define exp groups for parameter search
EXP_GROUPS = {'mnist': hu.cartesian_exp_group({
                           'dataset': 'mnist',
                           'model': 'mlp',
                           'max_epoch': 20,
                           'lr': [1e-3, 1e-4],
                           'batch_size': [32, 64]})}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs='+')
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument('-r', '--reset',  default=0, type=int)
    parser.add_argument('-ei', '--exp_id', default=None)
    parser.add_argument('-v', '--view_jupyter', default=None)
    parser.add_argument('-j', '--run_jobs', default=None)

    args = parser.parse_args()

    # Collect experiments
    # -------------------
    if args.exp_id is not None:
        # select one experiment
        savedir = os.path.join(args.savedir_base, args.exp_id)
        exp_dict = hu.load_json(os.path.join(savedir, 'exp_dict.json'))
        exp_list = [exp_dict]

    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += EXP_GROUPS[exp_group_name]

    # Run experiments or View them
    # ----------------------------
    if args.view_jupyter:
        # view results
        hj.view_jupyter(exp_list,
                        savedir_base=args.savedir_base,
                        fname='example.ipynb',
                        job_utils_path='results',
                        install_flag=False)

    elif args.run_jobs:
        # launch jobs
        from haven import haven_jobs as hj
        hj.run_exp_list_jobs(
            exp_list, savedir_base=args.savedir_base,
            workdir=os.path.dirname(os.path.realpath(__file__)))

    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict, savedir_base=args.savedir_base,
                     reset=args.reset)
