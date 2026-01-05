import argparse
import logging
import os
import random
import shutil
import time
from typing import Callable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import monai

from dataset import BaseDataSets

from monai.metrics import compute_dice, compute_iou, compute_hausdorff_distance
from metrics import compute_f1_rec_pre_acc

from model import DSCSegFormer_pretrained


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../mix_crack_3407_no_wrong.h5', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='Crack3238_lr00001_dicece_all_metrics_0',
                    help='experiment_name')  # every time you run, change the exp, to avoid overwriting the previous model
parser.add_argument('--max_epoch', type=int,
                    default=100, help='maximum number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=3407, help='random seed')
parser.add_argument('--num_classes', type=int, default=2,
                    help='output channel of network')


def softmax_to_onehot(softmax_output):
    pred = torch.argmax(softmax_output, dim=1)
    one_hot = F.one_hot(pred, num_classes=softmax_output.size(1)).permute(0, 3, 1, 2).float()
    return one_hot


def binary_label_to_onehot(labels):
    labels = labels.squeeze(1)
    onehot_labels = F.one_hot(labels.long(), num_classes=2).permute(0, 3, 1, 2).float()
    return onehot_labels


def train(args, snapshot_path, model, model_name):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_epoch = args.max_epoch

    # model_path = os.path.join(snapshot_path, '{}_best_model1.pth'.format(model_name))
    # model.load_state_dict(torch.load(model_path))
    model = model.to(dtype=torch.float)
    model = model.cuda()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    logging.info(f"Total number of parameters: {total_params}")
    # input_tensor = torch.randn(1, 6, 256, 256, device="cuda", dtype=torch.float)
    # flops = FlopCountAnalysis(model, input_tensor)
    # print(f"FLOPs: {flops.total()}")

    db_train = BaseDataSets(base_dir=args.root_path, split="train",
                            transform=transforms.Normalize(mean=[0.54415018, 0.53473719, 0.51293123],
                                                           std=[0.15626412, 0.16044922, 0.16692232]))  # 数值上还要改
    db_val = BaseDataSets(base_dir=args.root_path, split="val",
                          transform=transforms.Normalize(mean=[0.54415018, 0.53473719, 0.51293123],
                                                         std=[0.15626412, 0.16044922, 0.16692232]))

    trainloader = DataLoader(db_train, shuffle=True, num_workers=0, pin_memory=True, batch_size=args.batch_size)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=0)

    max_iterations = args.max_epoch * (len(db_train) // args.batch_size)

    model.train()

    # optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)

    # loss only calculates the crack (foreground) channel, not the background channel
    ce_loss = nn.BCELoss().to("cuda")
    dice_loss = monai.losses.DiceLoss(include_background=False).to("cuda")
    # fl_loss = FocalLoss(alpha=0.95, gamma=2).to("cuda")
    # dice_ce_loss = monai.losses.DiceCELoss(include_background=False, lambda_dice=1, lambda_ce=10).to("cuda")
    # dice_focal_loss = monai.losses.DiceFocalLoss(include_background=False, lambda_dice=1, lambda_focal=5).to("cuda")

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iterate_num = 0
    best_dice_performance = 0.0
    best_IoU_performance = 0.0

    for i_epoch in tqdm(range(max_epoch), ncols=70, desc="Epochs"):
        for i_batch, sampled_batch in enumerate(
                tqdm(trainloader, ncols=70, desc=f"Epoch {i_epoch + 1}/{max_epoch}", leave=False)):
            # if i_batch >= 1:
            #     break
            # sampled_batch split to labeled and unlabeled
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            label_batch = binary_label_to_onehot(label_batch)

            # Train Model 1
            prediction = model(image_batch)  # P1n, S1

            # prediction [b, 2, h, w]
            loss_dice = dice_loss(prediction, label_batch)  # loss s1
            loss_bce = ce_loss(prediction[:, 1], label_batch[:, 1])  # loss bce1
            # loss_dice_focal = dice_focal_loss(prediction, label_batch)
            # loss_dice_ce = dice_ce_loss(prediction, label_batch)
            supervised_loss = loss_dice + 10 * loss_bce  # dice loss 和 bce loss
            # supervised_loss = loss_dice


            loss = supervised_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # lr_ = base_lr * (1.0 - iterate_num / max_iterations) ** 3  # 0.9
            # lr_ = base_lr * (1.0 - iterate_num / max_iterations) ** 2  # pow 2
            lr_ = base_lr
            # lr_ = base_lr + 0.5 * (10 * base_lr - base_lr) * (1 + np.cos(np.pi * iterate_num / 250))  
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iterate_num += 1

            writer.add_scalar('info/lr', lr_, iterate_num)
            writer.add_scalar('info/total_loss', loss.item(), iterate_num)
            writer.add_scalar('info/loss_dice', loss_dice.item(), iterate_num)
            writer.add_scalar('info/loss_bce', loss_bce.item(), iterate_num)
            # writer.add_scalar('info/loss_focal', loss_fl.item(), iterate_num)

            # logging.info(
            #     'iteration %d : loss : %.5f, loss_dice: %.5f, loss_bce: %.5f' %
            #     (iterate_num, loss.item(), loss_dice.item(), loss_bce.item()))
            # logging.info(
            #     'iteration %d : loss : %.5f, loss_dice: %.5f,  loss_focal: %.5f' %
            #     (iterate_num, loss.item(), loss_dice.item(), loss_fl.item()))
            logging.info(
                'iteration %d : loss : %.5f' %
                (iterate_num, loss.item()))

        if i_epoch >= 50:
            # continue
            model.eval()
            val_dice_sum = 0.0
            val_IoU_sum = 0.0
            val_HD_sum = 0.0
            val_f1_sum = 0.0
            val_recall_sum = 0.0
            val_precision_sum = 0.0
            val_accuracy_sum = 0.0

            for i_batch_val, val_sampled_batch in enumerate(valloader):
                # if i_batch_val > 1:
                #     continue
                val_img_batch, val_label_batch = val_sampled_batch['image'], val_sampled_batch['label']
                val_img_batch, val_label_batch = val_img_batch.cuda(), val_label_batch.cuda()

                val_label_batch = binary_label_to_onehot(val_label_batch)
                val_prediction = softmax_to_onehot(model(val_img_batch))

                val_dice = compute_dice(val_prediction, val_label_batch, include_background=False, ignore_empty=False)
                val_IoU = compute_iou(val_prediction, val_label_batch, include_background=False, ignore_empty=False)
                val_HD = compute_hausdorff_distance(val_prediction, val_label_batch, include_background=False)
                val_f1, val_rec, val_pre, val_acc = compute_f1_rec_pre_acc(val_prediction, val_label_batch)

                # print(val_dice.item(), val_dice1.item())

                val_dice_sum += val_dice.item()
                val_IoU_sum += val_IoU.item()
                val_HD_sum += val_HD.item()
                val_f1_sum += val_f1.item()
                val_recall_sum += val_rec.item()
                val_precision_sum += val_pre.item()
                val_accuracy_sum += val_acc.item()

            dice_performance = val_dice_sum / len(valloader)
            IoU_performance = val_IoU_sum / len(valloader)
            HD_performance = val_HD_sum / len(valloader)
            f1_performance = val_f1_sum / len(valloader)
            recall_performance = val_recall_sum / len(valloader)
            precision_performance = val_precision_sum / len(valloader)
            accuracy_performance = val_accuracy_sum / len(valloader)

            if dice_performance > best_dice_performance:
                best_dice_performance = dice_performance
                best_IoU_performance = IoU_performance
                save_mode_path = os.path.join(snapshot_path,
                                              '1epoch_{}_dice_{}.pth'.format(
                                                  i_epoch, round(best_dice_performance, 4)))
                save_best = os.path.join(snapshot_path,
                                         '{}_best_model1.pth'.format(model_name))
                # torch.save(model1.state_dict(), save_mode_path)
                torch.save(model.state_dict(), save_best)

            writer.add_scalar('info/performance_dice', dice_performance, i_epoch)
            writer.add_scalar('info/performance_IoU', IoU_performance, i_epoch)
            writer.add_scalar('info/performance_HD', HD_performance, i_epoch)
            writer.add_scalar('info/performance_f1', f1_performance, i_epoch)
            writer.add_scalar('info/performance_recall', recall_performance, i_epoch)
            writer.add_scalar('info/performance_precision', precision_performance, i_epoch)
            writer.add_scalar('info/performance_accuracy', accuracy_performance, i_epoch)

            logging.info(
                'epoch %d : mean_dice : %f  best_dice : %f : mean_IoU : %f  best_IoU : %f ' %
                (i_epoch, dice_performance, best_dice_performance, IoU_performance, best_IoU_performance))

            model.train()

        # if iterate_num >= max_iterations:
        #     iterator.close()
        #     break
    writer.close()

    return print("Training Finished!")


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    num_classes = args.num_classes
    model_dict = {
        "DSCSegFormer_pretrained_pyramid_newcbam": lambda: DSCSegFormer_pretrained(n_channels=3),


    }

    for index, (model_name, model_) in enumerate(model_dict.items()):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        print(f"{model_name} is training...")
        model = model_()

        snapshot_path = "../model/{}_labeled/{}".format(
            args.exp, model_name)
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
        if os.path.exists(snapshot_path + '/code'):
            shutil.rmtree(snapshot_path + '/code')
        shutil.copytree('.', snapshot_path + '/code',
                        shutil.ignore_patterns(['.git', '__pycache__']))

        logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                            format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
        # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info(str(args))
        # logging.propagate = False
        train(args, snapshot_path, model, model_name)
        
        logger = logging.getLogger()
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
        logging.shutdown()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)
