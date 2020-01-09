"""MLP on MNIST example for Haven."""
import torch
import argparse
# import os
import pandas as pd
import torchvision
from torch.utils.data import DataLoader
import pprint
import os

import tqdm
from torch import nn
from torch.nn import functional as F
import sys

import exp_configs
from haven import haven_utils as hu
from haven import haven_results as hr
from haven import haven_chk as hc
from haven import haven_jupyter as hj


def trainval(exp_dict, savedir_base, reset):
    # ==================
    # bookkeepting stuff
    # ==================
    pprint.pprint(exp_dict)
    exp_id = hu.hash_dict(exp_dict)
    savedir = savedir_base + "/%s/" % exp_id
    if reset:
        hc.delete_and_backup_experiment(savedir)

    os.makedirs(savedir, exist_ok=True)
    hu.save_json(savedir+"exp_dict.json", exp_dict)
    print("Experiment saved in %s" % savedir)

    # ==================
    # Dataset
    # ==================
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize((0.5,), (0.5,))])

    # train set
    train_set = torchvision.datasets.MNIST(savedir_base,
                                           train=True,
                                           download=True,
                                           transform=transform)
    train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=exp_dict["batch_size"])

    # val set
    val_set = torchvision.datasets.MNIST(savedir_base, train=False,
                                         download=True, transform=transform)
    val_loader = DataLoader(val_set, shuffle=False,
                            batch_size=exp_dict["batch_size"])

    # ==================
    # Model
    # ==================
    model = MLP(n_classes=10).cuda()
    model.opt = torch.optim.Adam(model.parameters(), lr=exp_dict["lr"])
    
    model_path = savedir + "/model.pth"
    score_list_path = savedir + "/score_list.pkl"

    if os.path.exists(score_list_path):
        # resume experiment
        model.set_state_dict(hu.torch_load(model_path))
        score_list = hu.load_pkl(score_list_path)
        s_epoch = len(score_list)
    else:
        # restart experiment
        score_list = []
        s_epoch = 0

    # ==================
    # Train & Val
    # ==================
    print("Starting experiment at epoch %d" % s_epoch)
    for e in range(s_epoch, 100):
        score_dict = {}

        # Train the model
        score_dict.update(model.train_on_loader(train_loader))

        # Validate the model
        score_dict.update(model.val_on_loader(val_loader))
        score_dict["epoch"] = e

        # Add to score_list and save checkpoint
        score_list += [score_dict]

        # Report & Save
        score_df = pd.DataFrame(score_list)
        print("\n", score_df.tail()[["epoch", "train_loss", "val_acc"]], "\n")
        hu.torch_save(model_path, model.get_state_dict())
        hu.save_pkl(score_list_path, score_list)
        print("Checkpoint Saved: %s" % savedir_base)

class MLP(nn.Module):
    def __init__(self, input_size=784, n_classes=10):
        """Constructor."""
        super().__init__()

        self.input_size = input_size
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, 256)])
        self.output_layer = nn.Linear(256, n_classes)

    def forward(self, x):
        """Forward pass of one batch."""
        x = x.view(-1, self.input_size)
        out = x
        for layer in self.hidden_layers:
            Z = layer(out)
            out = F.relu(Z)
        logits = self.output_layer(out)

        return logits
    
    def get_state_dict(self):
        return {'model': self.state_dict(),
                'opt': self.opt.state_dict()} 

    def set_state_dict(self, state_dict):
        self.load_state_dict(state_dict['model'])
        self.opt.load_state_dict(state_dict['opt'])

    def train_on_loader(self, train_loader):
        """Train for one epoch."""
        self.train()
        loss_sum = 0.

        n_batches = len(train_loader)
        pbar = tqdm.tqdm(total=n_batches)
        for i, batch in enumerate(train_loader):
            loss_sum += float(self.train_on_batch(batch))

            pbar.set_description("Training loss: %.4f" % (loss_sum / (i + 1)))
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
        pbar = tqdm.tqdm(total=n_batches)
        for i, batch in enumerate(val_loader):
            gt_labels = batch[1]
            pred_labels = self.predict_on_batch(batch)

            se += float((pred_labels.cpu() == gt_labels).sum())
            n_samples += gt_labels.shape[0]

            pbar.set_description("Val score: %.4f" % (se / n_samples))
            pbar.update(1)

        pbar.close()
        acc = se / n_samples

        return {"val_acc": acc}

    def train_on_batch(self, batch):
        """Train for one batch."""
        images, labels = batch
        images, labels = images.cuda(), labels.cuda()

        self.opt.zero_grad()
        probs = F.log_softmax(self(images), dim=1)
        loss = F.nll_loss(probs, labels, reduction="mean")
        loss.backward()

        self.opt.step()

        return loss.item()

    def predict_on_batch(self, batch, **options):
        """Predict for one batch."""
        images, labels = batch
        images = images.cuda()
        probs = F.log_softmax(self(images), dim=1)

        return probs.argmax(dim=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--exp_group_list', nargs="+")
    parser.add_argument('-sb', '--savedir_base', required=True)
    parser.add_argument("-r", "--reset",  default=0, type=int)
    parser.add_argument("-ei", "--exp_id", default=None)
    parser.add_argument("-v", "--view_experiments", default=None)
    parser.add_argument("-j", "--create_jupyter", default=None)

    args = parser.parse_args()

    # =====================
    # Collect experiments
    # =====================
    if args.exp_id is not None:
        # select one experiment
        savedir = args.savedir_base + "/%s/" % args.exp_id
        exp_dict = hu.load_json(savedir+"/exp_dict.json")        
        
        exp_list = [exp_dict]
        
    else:
        # select exp group
        exp_list = []
        for exp_group_name in args.exp_group_list:
            exp_list += exp_configs.EXP_GROUPS[exp_group_name]

    # =====================
    # Run experiments or View them
    # =====================
    if args.view_experiments:
        # view experiments
        hr.view_experiments(exp_list, savedir_base=args.savedir_base)

    elif args.create_jupyter:
        # view experiments
        hj.create_jupyter(args.exp_group_list, 
                        savedir_base=args.savedir_base, 
                       workdir=os.path.dirname(__file__),
                       legend_list=['lr','batch_size'],
                       score_list=['train_loss', 'val_acc'])

    else:
        # run experiments
        for exp_dict in exp_list:
            # do trainval
            trainval(exp_dict=exp_dict,
                    savedir_base=args.savedir_base,
                    reset=args.reset)

