import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisionLoss(torch.nn.Module):

    def __init__(self):
        super(SupervisionLoss, self).__init__()

    # def forward(self, *x):
    #     y, y_hat, extra_args = x
    #     return self.forward(y, y_hat, extra_args)

    def forward(self, y, y_hat, extra_args):
        raise NotImplementedError

class TopkCrossEntropy(SupervisionLoss):

    def __init__(self, top_k=0.5, weight=None, size_average=None,
                 ignore_index=-100, reduce=None, reduction='none'):
        super(TopkCrossEntropy, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.top_k = top_k
        self.loss = torch.nn.NLLLoss(weight=self.weight,
                                     ignore_index=self.ignore_index, reduction='none')

    def forward(self, y, y_hat, extra_args):
        y = y.long()
        loss = self.loss(F.log_softmax(y_hat, dim=1), y)
        loss = loss.view(-1)
        if self.top_k == 1:
            return torch.mean(loss)
        else:
            y_exclude = extra_args.get('b', None)
            _, idxs = torch.topk(loss, int(self.top_k * loss.size()[0]))
            if y_exclude is not None and y_exclude.sum() > 0:
                exclude_idxs = y_exclude.view(-1).nonzero().view(-1)
            else:
                exclude_idxs = None
            if exclude_idxs is not None:
                temp_indices = torch.ones_like(idxs).bool()
                for elem in exclude_idxs:
                    temp_indices = temp_indices & (idxs != elem)
                idxs = idxs[temp_indices]
            valid_loss = loss[idxs]
            return torch.mean(valid_loss)