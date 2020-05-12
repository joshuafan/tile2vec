import numpy as np
import time
import torch
from torch.autograd import Variable
from datasets import triplet_dataloader

def prep_triplets(triplets, cuda):
    """
    Takes a batch of triplets and converts them into Pytorch variables 
    and puts them on GPU if available.
    """
    a, n, d = (Variable(triplets['anchor']), Variable(triplets['neighbor']), Variable(triplets['distant']))
    if cuda:
    	a, n, d = (a.cuda(), n.cuda(), d.cuda())
    return (a, n, d)

def train_triplet_epoch(model, cuda, dataloader, optimizer, epoch, is_train=True, margin=1,
    l2=0, print_every=100, t0=None):
    """
    Trains a model for one epoch using the provided dataloader.
    """
    if is_train:
        model.train()
    else:
        model.eval()

    if t0 is None:
        t0 = time.time()
    sum_loss, sum_l_n, sum_l_d, sum_l_nd, sum_l_recon = (0, 0, 0, 0, 0)
    n_train, n_batches = len(dataloader.dataset), len(dataloader)
    print_sum_loss = 0
    for idx, triplets in enumerate(dataloader):
        p, n, d = prep_triplets(triplets, cuda)

        #print('p shape', p[0].shape, 'band means', torch.mean(p[0], dim=(1,2)))
        #print('n band means', torch.mean(n[0], dim=(1,2)))
        #print('d band means', torch.mean(d[0], dim=(1,2)))
        #print('p random pixel', p[0, :, 5, 5])
        optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            loss, l_n, l_d, l_nd, l_recon = model.loss(p, n, d, margin=margin, l2=l2)
        if is_train:
            loss.backward()
            #print('Decoder 2 grad', model.decoder2.weight.grad)
            optimizer.step()
        #print("loss", loss)
        sum_loss += loss.item()  #data[0]
        sum_l_n += l_n.item()  # data[0]
        sum_l_d += l_d.item()  # data[0]
        sum_l_nd += l_nd.item()  # data[0]
        sum_l_recon += l_recon.item()
        if (idx + 1) * dataloader.batch_size % print_every == 0:
            print_avg_loss = (sum_loss - print_sum_loss) / (
                print_every / dataloader.batch_size)
            print('Epoch {}: [{}/{} ({:0.0f}%)], Avg loss: {:0.4f}'.format(
                epoch, (idx + 1) * dataloader.batch_size, n_train,
                100 * (idx + 1) / n_batches, print_avg_loss))
            print_sum_loss = sum_loss
    avg_loss = sum_loss / n_batches
    avg_l_n = sum_l_n / n_batches
    avg_l_d = sum_l_d / n_batches
    avg_l_nd = sum_l_nd / n_batches
    avg_l_recon = sum_l_recon / n_batches
    print('Finished epoch {}: {:0.3f}s'.format(epoch, time.time()-t0))
    print('  Average loss: {:0.4f}'.format(avg_loss))
    print('  Average l_n: {:0.4f}'.format(avg_l_n))
    print('  Average l_d: {:0.4f}'.format(avg_l_d))
    print('  Average l_nd: {:0.4f}\n'.format(avg_l_nd))
    print('  Average l_recon: {:0.4f}\n'.format(avg_l_recon))
    return (avg_loss, avg_l_n, avg_l_d, avg_l_nd)
