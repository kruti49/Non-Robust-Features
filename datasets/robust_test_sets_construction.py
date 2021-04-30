from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR
import torch as ch
# We use cox (http://github.com/MadryLab/cox) to log, store and analyze
# results. Read more at https//cox.readthedocs.io.
from cox.utils import Parameters
import cox.store

import os
os.environ['NOTEBOOK_MODE'] = '1'
import sys
import numpy as np
import seaborn as sns
from scipy import stats
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
from robustness import model_utils, datasets
from robustness.tools.vis_tools import show_image_row, show_image_column
from torchvision.utils import save_image
%matplotlib inline

DATA_PATH_DICT = {'CIFAR': './cifar10'}
DATA = 'CIFAR' # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']
NUM_WORKERS = 4
NOISE_SCALE = 20
OUT_DIR = './outputs/'
BATCH_SIZE = 200

DATA_SHAPE = 32 if DATA == 'CIFAR' else 224 # Image size (fixed for dataset)
REPRESENTATION_SIZE = 2048 # Size of representation vector (fixed for model)

# Hard-coded dataset, architecture, batch size, workers
ds = CIFAR('/tmp/')
model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=ds)
train_loader, test_loader = ds.make_loaders(batch_size=BATCH_SIZE, workers=NUM_WORKERS)

# Create a cox store for logging
out_store = cox.store.Store(OUT_DIR)

# Hard-coded base parameters
train_kwargs = {
    'out_dir': "./outputs",
    'adv_train': 1,
    'constraint': '2',
    'eps': 0.5,
    'attack_lr': 0.1,
    'attack_steps': 100,
    'epochs': 100
}
train_args = Parameters(train_kwargs)

# Fill missing parameters
train_args = defaults.check_and_fill_args(train_args,
                        defaults.TRAINING_ARGS, CIFAR)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.PGD_ARGS, CIFAR)

# Train a model
train.train_model(train_args, model, (train_loader, test_loader), store=out_store)

# Load dataset
dataset_function = getattr(datasets, DATA)
dataset = dataset_function(DATA_PATH_DICT[DATA])
_, test_loader = dataset.make_loaders(workers=NUM_WORKERS, 
                                      batch_size=BATCH_SIZE, 
                                      data_aug=False)
data_iterator = enumerate(test_loader)

#Load model from checkpoint
model_kwargs = {
    'arch': 'resnet50',
    'dataset': dataset,
    'resume_path': f'./checkpoint/robustifiedckpt.pt'
}

model, _ = model_utils.make_and_restore_model(**model_kwargs)
model.eval()

# Custom loss for inversion
def inversion_loss(model, inp, targ):
    _, rep = model(inp, with_latent=True, fake_relu=True)
    loss = ch.div(ch.norm(rep - targ, dim=1), ch.norm(targ, dim=1))
    return loss, None

# PGD parameters
kwargs = {
    'custom_loss': inversion_loss,
    'constraint':'2',
    'eps':0.125,#Change from [0,1/16,1/4,1/2,1] to generate R - Robustified test sets
    'step_size': 0.05,
    'iterations': 400, 
    'do_tqdm': True,
    'targeted': True,
    'use_best': False
}

for batch_ize,(img,targt) in enumerate(test_loader):

    im,targ = img.cuda(),targt.cuda()

    with ch.no_grad():
        (_, rep), _ = model(im.cuda(), with_latent=True) # Corresponding representation

    im_n = ch.randn_like(im) / NOISE_SCALE + 0.5 # Seed for inversion (x_0)

    target, xadv = model(im_n.cuda(), rep.clone(), make_adv=True, **kwargs) # Image inversion using PGD

    # Visualize inversion
    show_image_row([im.cpu(), im_n.cpu(), xadv.detach().cpu()], 
               ["Original", r"Seed ($x_0$)", "Result"], 
               fontsize=22)
			   
    save_image(xadv.cpu(),"./R_test_sets/cifar"+str(batch_ize)+".png")
