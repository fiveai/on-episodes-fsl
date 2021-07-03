import torch
import torch.nn.functional as F
import numpy as np


class FewShotNCALoss(torch.nn.Module):
    def __init__(
        self,
        classes,
        temperature,
        batch_size,
        frac_negative_samples=1,
        frac_positive_samples=1,
    ):
        super(FewShotNCALoss, self).__init__()
        self.temperature = torch.tensor(float(temperature), requires_grad=True).cuda()
        self.cls = classes
        self.batch_size = batch_size
        self.frac_negative_samples = frac_negative_samples
        self.frac_positive_samples = frac_positive_samples

    def forward(self, pred, target):
        n, d = pred.shape
        # identity matrix needed for masking matrix
        self.eye = torch.eye(target.shape[0]).cuda()

        # compute distance
        p_norm = torch.pow(torch.cdist(pred, pred), 2)
        # lower bound distances to avoid NaN errors
        p_norm[p_norm < 1e-10] = 1e-10
        dist = torch.exp(-1 * p_norm / self.temperature).cuda()


        # create matrix identifying all positive pairs
        bool_matrix = target[:, None] == target[:, None].T
        # substracting identity matrix removes positive pair with itself
        positives_matrix = (
            torch.tensor(bool_matrix, dtype=torch.int16).cuda() - self.eye
        ).cuda()
        # negative matrix is the opposite using ~ as not operator
        negatives_matrix = torch.tensor(~bool_matrix, dtype=torch.int16).cuda()

        # sampling random elements for the negatives
        if self.frac_negative_samples < 1:
            # create a new mask
            mask = torch.zeros(n, n).cuda()

            negatives_idx = (negatives_matrix == 1).nonzero()

            n_to_sample = int(negatives_idx.shape[0] * self.frac_negative_samples)

            choice = np.random.choice(
                negatives_idx.shape[0], size=n_to_sample, replace=False
            )

            choice = negatives_idx[choice, :]

            mask[choice[:, 0], choice[:, 1]] = 1

            # create random negatives mask
            negatives_matrix = negatives_matrix * mask
            denominators = torch.sum(dist * negatives_matrix, axis=0).cuda()
        else:
            denominators = torch.sum(dist * negatives_matrix, axis=0)

        if self.frac_positive_samples < 1:
            # create a new mask
            mask = torch.zeros(n, n).cuda()

            positives_idx = (positives_matrix == 1).nonzero()

            n_to_sample = int(positives_idx.shape[0] * self.frac_positive_samples)

            choice = np.random.choice(
                positives_idx.shape[0], size=n_to_sample, replace=False
            )

            choice = positives_idx[choice, :]

            mask[choice[:, 0], choice[:, 1]] = 1

            positives_matrix = positives_matrix * mask
            numerators = torch.sum(dist * positives_matrix, axis=0).cuda()
        else:
            numerators = torch.sum(dist * positives_matrix, axis=0)

        # avoiding nan errors
        denominators[denominators < 1e-10] = 1e-10
        frac = numerators / (numerators + denominators)

        loss = -1 * torch.sum(torch.log(frac[frac >= 1e-10])) / n

        return loss


class SupervisedContrastiveLoss(torch.nn.Module):
    def __init__(self, classes, temperature, batch_size, n_samples=0, frac_hard_samples=0):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cls = classes
        self.batch_size = batch_size
        self.n_samples = n_samples

    def forward(self, pred, target):
        n, d = pred.shape
        # identity matrix needed
        self.eye = torch.eye(target.shape[0]).cuda()

        x1 = pred.unsqueeze(1).expand(n, n, d)
        x2 = pred.unsqueeze(0).expand(n, n, d)

        # compute pairwise distances between all points
        p_norm = F.cosine_similarity(x1, x2, dim=2)

        # lower bound distances to avoid NaN errors
        p_norm[p_norm < 1e-10] = 1e-10
        dist = torch.exp(p_norm / self.temperature).cuda()

        # create matrix identifying all positive pairs
        bool_matrix = target[:, None] == target[:, None].T
        # substracting identity matrix removes positive pair with itself
        positives_matrix = (
            torch.tensor(bool_matrix, dtype=torch.int32).cuda() - self.eye
        ).cuda()
        # negative matrix is the opposite
        negatives_matrix = torch.tensor(~bool_matrix, dtype=torch.int32).cuda()

        denominators = torch.sum(dist * negatives_matrix, axis=0)

        # compute numerators and denominators for NCA
        numerators = (dist * positives_matrix).cuda()

        # avoiding nan errors
        denominators[denominators < 1e-10] = 1e-10
        frac = numerators / denominators

        loss = (
            -1
            * torch.sum(torch.sum(torch.log(frac[frac >= 1e-10]), axis=0))
            / self.batch_size
        )

        return loss
