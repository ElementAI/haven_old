import torch
from torch import nn
from torch.nn import functional as F


def get_model(model_name, use_cuda=True):
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if model_name == 'mlp':
        return MLP(device=device).to(device)


class MLP(nn.Module):
    def __init__(self, input_size=784, n_classes=10, device="cpu"):
        """Constructor."""
        super().__init__()

        self.input_size = input_size
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, 256)])
        self.output_layer = nn.Linear(256, n_classes)

        self.opt = torch.optim.SGD(self.parameters(), lr=1e-3)
        self.device = device

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

        for i, batch in enumerate(train_loader):
            loss_sum += float(self.train_on_batch(batch))

            if i % (n_batches//10) == 0:
                print("   Iteration: %04d/%04d - Training loss: %.4f" %
                      (i, len(train_loader), loss_sum / (i + 1)))

        loss = loss_sum / n_batches

        return {"train_loss": loss}

    @torch.no_grad()
    def val_on_loader(self, val_loader):
        """Validate the model."""
        self.eval()
        se = 0.
        n_samples = 0

        n_batches = len(val_loader)

        for i, batch in enumerate(val_loader):
            gt_labels = batch[1]
            pred_labels = self.predict_on_batch(batch)

            se += float((pred_labels.cpu() == gt_labels).sum())
            n_samples += gt_labels.shape[0]

            if i % (n_batches//10) == 0:
                print("   Iteration: %04d/%04d - Val accuracy: %.4f" %
                      (i, len(val_loader), se / n_samples))

        acc = se / n_samples

        return {"val_acc": acc}

    def train_on_batch(self, batch):
        """Train for one batch."""
        images, labels = batch
        images = images.to(self.device)
        labels = labels.to(self.device)

        self.opt.zero_grad()
        probs = F.log_softmax(self(images), dim=1)
        loss = F.nll_loss(probs, labels, reduction="mean")
        loss.backward()

        self.opt.step()

        return loss.item()

    def predict_on_batch(self, batch, **options):
        """Predict for one batch."""
        images, labels = batch
        images = images.to(self.device)
        probs = F.log_softmax(self(images), dim=1)

        return probs.argmax(dim=1)
