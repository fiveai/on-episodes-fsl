import numpy as np
import torch
import tqdm


class BestAccuracySlots:
    """
    Class to keep track of best scores on different normalisation
    """

    __slots__ = (
        "cl2n_1shot",
        "cl2n_5shot",
    )

    def __init__(self):
        self.cl2n_1shot = -1
        self.cl2n_5shot = -1

    def update(self, shot1info, shot5info):
        shot1_cl2n_acc = shot1info[0]
        shot5_cl2n_acc = shot5info[0]

        # choose best epoch according to CL2N performance, update everything
        is_best1 = shot1_cl2n_acc > self.cl2n_1shot
        is_best5 = shot5_cl2n_acc > self.cl2n_5shot
        self.cl2n_1shot = max(self.cl2n_1shot, shot1_cl2n_acc)
        self.cl2n_5shot = max(self.cl2n_5shot, shot5_cl2n_acc)
        return is_best1, is_best5


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def warp_tqdm(data_loader, args):
    """
    warp tqdm around a dataloader
    """
    if args.disable_tqdm:
        tqdm_loader = data_loader
    else:
        tqdm_loader = tqdm.tqdm(data_loader, total=len(data_loader))
    return tqdm_loader
