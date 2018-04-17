from torch.nn import functional


def squared_error(prediction, target):
    mask = prediction - target
    mask = mask.abs() <= 0.4
    loss = functional.mse_loss(prediction, target, reduce=False)
    loss.masked_fill_(mask, 0)
    return loss.sum()
