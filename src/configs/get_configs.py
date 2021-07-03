import torch
import pickle
from torch.optim.lr_scheduler import MultiStepLR, StepLR, CosineAnnealingLR
import torch.nn.functional as F
import src.datasets as datasets


def get_dataloader(split, args, aug=False, shuffle=True, out_name=False, sample=None):
    """
    returns the dataloader, which either loads the training, validation or test set_description
    :param split (str): load the train, val or test
    :param args: args which contains variables needed
    :param aug (bool): augment the data using datasets.with_augment
    :param shuffle (bool): shuffle instances in the dataloader
    :param out_name (bool): returns image name in the batch
    :param sample (list): list of meta_val interation, meta_val way, meta_val shot,
                          and query. samples few shot method

    :return DataLoader:
    """
    # sample: iter, way, shot, query
    if aug:
        transform = datasets.with_augment(
            84, disable_random_resize=args.disable_random_resize)
    else:
        transform = datasets.without_augment(84, enlarge=args.enlarge)

    sets = datasets.DatasetFolder(
        args.data, args.split_dir, split, transform, out_name=out_name)
    if sample is not None:
        # this is for prototypical network training
        sampler = datasets.CategoriesSampler(sets.labels,
                                             args.episode_no_replacement_sampling,
                                             *sample)

        loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler,
                                             num_workers=args.workers, pin_memory=True)
    else:
        # this is for normal sampling
        if args.replacement_sampling:
            sampler = datasets.ReplacementSampler(sets.labels, args.proto_train_iter,
                                                  args.batch_size)

            loader = torch.utils.data.DataLoader(sets,
                                                 batch_sampler=sampler,
                                                 num_workers=args.workers,
                                                 pin_memory=True)
        else:
            loader = torch.utils.data.DataLoader(sets,
                                                 batch_size=args.batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=args.workers,
                                                 pin_memory=True)
    return loader


def get_metric(metric_type, args):
    """
    :param metric_type (str): choose which metric function to use (cosine, euclidean, l1, l2)
    :param args: args which contains variables needed
    :return metric function (callable function):
    """
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


def get_scheduler(batches, optimizer, args):
    """
    cosine will change learning rate every iteration, others change learning rate every epoch
    :param batches: the number of iterations in each epochs
    :return: scheduler
    """

    SCHEDULER = {'step': StepLR(optimizer, args.lr_stepsize, args.lr_gamma),
                 'multi_step': MultiStepLR(optimizer,
                                           milestones=[round(float(x)*args.epochs) for x in args.lr_milestones.split(',')],
                                           gamma=args.lr_gamma),
                 'cosine': CosineAnnealingLR(optimizer, batches * args.epochs, eta_min=1e-9)}
    return SCHEDULER[args.scheduler]


def save_pickle(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def get_optimizer(module, args):
    OPTIMIZER = {'SGD': torch.optim.SGD(module.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay,
                                        nesterov=args.nesterov),
                 'Adam': torch.optim.Adam(module.parameters(), lr=args.lr)}
    return OPTIMIZER[args.optimizer]
