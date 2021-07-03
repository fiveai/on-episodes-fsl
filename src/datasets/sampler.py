import numpy as np
import torch
from torch.utils.data import Sampler
import random
import copy

__all__ = ['CategoriesSampler', 'ReplacementSampler']


class CategoriesSampler(Sampler):

    def __init__(self, label, replacement, n_iter, n_way, n_shot, n_query):

        self.n_iter = n_iter
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.replacement = replacement

        label = np.array(label)
        unique = np.unique(label)
        unique = np.sort(unique)

        self.m_ind = []
        self.labels = unique
        # dictionary to keep track of which images belong to which class
        self.class2imgs = {}

        for i in unique:
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
            self.class2imgs[i] = list(ind.numpy())


    def __len__(self):
        return self.n_iter

    def __iter__(self):
        if self.replacement:
            for i in range(self.n_iter):
                batch_gallery = []
                batch_query = []
                classes = torch.randperm(len(self.m_ind))[:self.n_way]
                for c in classes:
                    l = self.m_ind[c.item()]
                    pos = torch.randperm(l.size()[0])
                    batch_gallery.append(l[pos[:self.n_shot]])
                    batch_query.append(l[pos[self.n_shot:self.n_shot + self.n_query]])
                batch = torch.cat(batch_gallery + batch_query)
                yield batch

        else:
            n_to_sample = (self.n_query + self.n_shot)
            batch_size = self.n_way*(self.n_query + self.n_shot)

            remaining_classes = list(self.labels)

            copy_class2imgs = copy.deepcopy(self.class2imgs)

            while len(remaining_classes) > self.n_way - 1:
                # randomly select classes
                classes = random.sample(remaining_classes, self.n_way)

                batch_gallery = []
                batch_query = []

                # construct the batch
                for c in classes:
                    # sample correct numbers
                    l = random.sample(copy_class2imgs[c], n_to_sample)
                    batch_gallery.append(torch.tensor(l[:self.n_shot], dtype=torch.int32))
                    batch_query.append(torch.tensor(l[self.n_shot:self.n_shot + self.n_query],
                                                    dtype=torch.int32))

                    # remove values if used (sampling without replacement)
                    for value in l:
                        copy_class2imgs[c].remove(value)

                    # if not enough elements remain,
                    # remove key from dictionary and remaining classes
                    if len(copy_class2imgs[c]) < n_to_sample:
                        del copy_class2imgs[c]
                        remaining_classes.remove(c)

                batch = torch.cat(batch_gallery + batch_query)
                yield batch

class ReplacementSampler(Sampler):
    """
    Sampler for a dataloader that samples batches such that between different
    batches the same elements can occur. For every batch, all training elements
    can be sampled, whereas during traditional batch sampling all elements must
    occur once.
    """
    def __init__(self, label, n_iter, batch_size):
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.label = label

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            # we set replace=False here. This means that within batches
            # there are no duplicates, however between batches there still can
            # be (hence it's still replacement sampling between batches)
            np_idx = np.random.choice(len(self.label), self.batch_size, replace=False)
            torch_idx = torch.from_numpy(np_idx)
            yield torch_idx
