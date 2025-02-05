import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import torch
import torchvision as tv
import os
from transformer_flow import Model
import utils
import pathlib
utils.set_random_seed(100)
notebook_output_path = pathlib.Path('runs/notebook')

import yaml

from ignite.contrib import metrics
# from sklearn.metrics import roc_auc_score, auc

import constants as const
import dataset
import utils
import random
import numpy as np
import wandb
import torch.nn.functional as F
import wandb



def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    dataset_name = 'mvtec'
    num_classes = 0
    img_size = 256
    channel_size = 3

    # we use a small model for fast demonstration, increase the model size for better results
    patch_size = 16
    channels = 128
    blocks = 4
    layers_per_block = 4
    # try different noise levels to see its effect
    noise_std = 0.1

    lr = 1e-3
    # increase epochs for better results
    epochs = 500
    sample_freq = 10

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'  # if on mac
    else:
        device = 'cpu'  # if mps not available

    ##########################
    if args.wandb_enable:
        wandb.init(
            # set the wandb project where this run will be logged
            project="tarflow",
            # track hyperparameters and run metadata
            config={
                "num_classes": num_classes,
                "dataset_name": dataset_name,
                "epochs": epochs,
                "batch_size": const.BATCH_SIZE,
                "img_size": img_size,
                "channel_size": channel_size,
                "patch_size": patch_size,
                "channels": channels,
                "blocks": blocks,
                "layers_per_block": layers_per_block,
                "lr_start": lr,
                "device": device,
            }
        )

    os.makedirs(const.CHECKPOINT_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(
        const.CHECKPOINT_DIR, "exp%d" % len(os.listdir(const.CHECKPOINT_DIR))
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    # config = yaml.safe_load(open(args.config, "r"))
    train_dataset = dataset.MVTecDataset(root=args.data, category=args.category, input_size=img_size, is_train=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=const.BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True,)

    model = Model(in_channels=channel_size, img_size=img_size, patch_size=patch_size,
                  channels=channels, num_blocks=blocks, layers_per_block=layers_per_block,
                  num_classes=num_classes).to(device)
    # model.cuda()

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.95), lr=lr, weight_decay=1e-4)
    lr_schedule = utils.CosineLRSchedule(optimizer, len(train_dataloader), epochs * len(train_dataloader), 1e-6, lr)

    model_name = f'{patch_size}_{channels}_{blocks}_{layers_per_block}_{noise_std:.2f}'
    sample_dir = notebook_output_path / f'{dataset_name}_samples_{model_name}'
    ckpt_file = notebook_output_path / f'{dataset_name}_model_{model_name}.pth'
    sample_dir.mkdir(exist_ok=True, parents=True)

    for epoch in range(epochs):
        # model.train()
        losses = 0
        # loss_meter = utils.AverageMeter()
        for x in train_dataloader:
            x = x.to(device)
            eps = noise_std * torch.randn_like(x)
            x = x + eps
            # y = y.to(device)
            optimizer.zero_grad()
            z, outputs, logdets = model(x, None)
            loss = model.get_loss(z, logdets)
            loss.backward()
            optimizer.step()
            lr_schedule.step()
            losses += loss.item()

        if args.wandb_enable:
            wandb.log({"lr": optimizer.param_groups[0]['lr'], "epoch": epoch,
                       "loss": losses / len(train_dataloader)})
        print(f"epoch {epoch} lr {optimizer.param_groups[0]['lr']:.6f} loss {losses / len(train_dataloader):.4f}")
        print('layer norms', ' '.join([f'{z.pow(2).mean():.4f}' for z in outputs]))

        eval_one_epoch(model, args.wandb_enable, img_size, patch_size)
    if args.wandb_enable:
        wandb.finish()
        # if (epoch + 1) % sample_freq == 0:
        #     with torch.no_grad():
        #         samples = model.reverse(fixed_noise, fixed_y)
        #     tv.utils.save_image(samples, sample_dir / f'samples_{epoch:03d}.png', normalize=True, nrow=10)
        #     tv.utils.save_image(model.unpatchify(z[:100]), sample_dir / f'latent_{epoch:03d}.png', normalize=True,
        #                         nrow=10)
        #     print('sampling complete')
    # torch.save(model.state_dict(), ckpt_file)


def eval_one_epoch(model, wandb_enable, img_size, patch_size):
    num_classes = 0
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'  # if on mac
    else:
        device = 'cpu'  # if mps not available
    # print(f'using device {device}')

    # we use a small model for fast demonstration, increase the model size for better results
    assert img_size % patch_size == 0
    num_of_token = int(img_size/patch_size)
    # try different noise levels to see its effect

    test_dataset = dataset.MVTecDataset(root=args.data, category=args.category, input_size=img_size, is_train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=const.BATCH_SIZE, shuffle=False,
                                                  num_workers=4, drop_last=False)

    auroc_metric = metrics.ROC_AUC()
    for lvl, (x, targets) in enumerate(test_dataloader):
        x = x.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            z, outputs, logdets = model(x, None)
            log_prob = -torch.mean(z ** 2, dim=2, keepdim=True) * 0.5
            log_prob_rsp = log_prob.view(x.shape[0], num_of_token, num_of_token, 1).permute(0, 3, 1, 2)
            prob = torch.exp(log_prob_rsp)
            a_map = F.interpolate(
                -prob,
                size=[img_size, img_size],
                mode="bilinear",
                align_corners=False,
            )

            res = a_map.cpu().detach()
            res = res.flatten()
            targets = targets.flatten().to(torch.bool)
            auroc_metric.update((res, targets))
            # loc_auroc = roc_auc_score(targets, res) * 100
            # det_auroc = roc_auc_score(gt_label, res) * 100
            # auroc_metric.update((res, targets))
    auroc = auroc_metric.compute()
    if wandb_enable:
        wandb.log({"auroc": auroc})

    print("AUROC: {}".format(auroc))

def evaluate(args):
    num_classes = 0
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'  # if on mac
    else:
        device = 'cpu'  # if mps not available
    # print(f'using device {device}')

    img_size = 256
    channel_size = 3

    # we use a small model for fast demonstration, increase the model size for better results
    patch_size = 16
    channels = 128
    blocks = 4
    layers_per_block = 4
    # try different noise levels to see its effect
    noise_std = 0.1
    # batch_size = 256

    test_dataset = dataset.MVTecDataset(root=args.data, category=args.category, input_size=img_size, is_train=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=const.BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)

    # transform = tv.transforms.Compose([
    #     tv.transforms.Resize((img_size, img_size)),
    #     tv.transforms.ToTensor(),
    #     tv.transforms.Normalize((0.5,), (0.5,))
    # ])

    model = Model(in_channels=channel_size, img_size=img_size, patch_size=patch_size,
                  channels=channels, num_blocks=blocks, layers_per_block=layers_per_block,
                  num_classes=num_classes).to(device)

    # now we can also evaluate the model by turning it into a classifier with Bayes rule, p(y|x) = p(y)p(x|y)/p(x)
    # data = tv.datasets.MNIST('.', transform=transform, train=False, download=False)
    # data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False)
    # num_correct = 0
    # num_examples = 0

    auroc_metric = metrics.ROC_AUC()
    # gt_label_list.extend(t2np(label))
    # gt_label = np.asarray(gt_label_list, dtype=np.bool_)
    output_lists = []
    target_lists = []
    for lvl, (x, targets) in enumerate(test_dataloader):
        x = x.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            z, outputs, logdets = model(x, None)
            log_prob = -torch.mean(z**2, dim=2, keepdim=True) * 0.5
            log_prob_rsp = log_prob.view(x.shape[0], patch_size, patch_size, 1).permute(0, 3, 1, 2)
            prob = torch.exp(log_prob_rsp)
            a_map = F.interpolate(
                -prob,
                size=[img_size, img_size],
                mode="bilinear",
                align_corners=False,
            )

            # output_lists.append(a_map)
            # target_lists.append(targets)

            res = a_map.cpu().detach()
            # targets = targets.cpu().detach()
            res = res.flatten()
            targets = targets.flatten().to(torch.bool)
            auroc_metric.update((res, targets))
            # loc_auroc = roc_auc_score(targets, res) * 100

            # det_auroc = roc_auc_score(gt_label, res) * 100
            # auroc_metric.update((res, targets))
        # output_tensor = torch.cat(output_lists, dim=0).cpu().detach().flatten()
        # target_tensor = torch.cat(target_lists, dim=0).cpu().detach().flatten()
    auroc = auroc_metric.compute()
    print("AUROC: {}".format(auroc))



            # losses = 0.5 * z.pow(2).mean(dim=[1, 2]) - logdets  # keep the batch dimension
            # pred = losses.reshape(num_classes, y.size(0)).argmin(dim=0)
        # num_correct += (pred == y).sum()
        # num_examples += y.size(0)
    # print(f'Accuracy %{100 * num_correct / num_examples:.2f}')


def parse_args():
    parser = argparse.ArgumentParser(description="Train FastFlow on MVTec-AD dataset")
    parser.add_argument(
        "-cfg", "--config", type=str, required=True, help="path to config file"
    )
    parser.add_argument("--data", type=str, required=True, help="path to mvtec folder")
    parser.add_argument('--dataset', default='mvtec', type=str,
                        choices=['mvtec', 'visa', 'miad'], help='dataset name')
    parser.add_argument(
        "-cat",
        "--category",
        type=str,
        choices=const.MVTEC_CATEGORIES,
        required=True,
        help="category name in mvtec",
    )
    parser.add_argument("--eval", action="store_true", help="run eval only")
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="path to load checkpoint"
    )
    # parser.add_argument('--attn_enable', action='store_true', default=False,
    #                     help='use attention flow or not.')
    # parser.add_argument('--attn_rsp', action='store_true', default=False,
    #                     help='use attention reshape or not.')
    parser.add_argument('--wandb_enable', action='store_true', default=False,
                     help='use attention reshape or not.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':


    # train_local()
    # eval_local()
    args = parse_args()
    # init_seeds(seed=4396)
    if args.eval:
        evaluate(args)
    else:
        train(args)
