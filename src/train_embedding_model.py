import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torch.optim as optim

from .datasets import triplet_dataloader
from .sample_tiles import get_triplet_imgs, get_triplet_tiles
from .tilenet import make_tilenet
from .training import train_triplet_epoch

DATASET_DIR = "datasets/dataset_2018-08-01"
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
TIF_IMAGE_DIR = "../../RemoteSensing/datasets/LandsatReflectance/2018-08-01"
TILE_DATASET_DIR = "../../RemoteSensing/datasets/tile2vec_tiles_2018-08-01"
MODEL_DIR = "../../RemoteSensing/models/tile2vec"
RGB_BANDS = [1, 2, 3]
img_triplets = get_triplet_imgs(TIF_IMAGE_DIR, n_triplets=20)
tiles = get_triplet_tiles(TILE_DATASET_DIR,
                          img_triplets,
                          tile_size=10,
                          neighborhood=100,
                          save=True,
                          verbose=True)

# Visualize
tile_dir = '../data/example_tiles/'
n_triplets = 2
plt.rcParams['figure.figsize'] = (12, 4)
for i in range(n_triplets):
    tile = np.load(os.path.join(tile_dir, str(i) + 'anchor.npy'))
    neighbor = np.load(os.path.join(tile_dir, str(i) + 'neighbor.npy'))
    distant = np.load(os.path.join(tile_dir, str(i) + 'distant.npy'))

    vmin = np.array([tile, neighbor, distant]).min()
    vmax = np.array([tile, neighbor, distant]).max()

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(tile[RGB_BANDS, :, :])
    plt.title('Anchor ' + str(i))
    plt.subplot(1, 3, 2)
    plt.imshow(neighbor[RGB_BANDS, :, :])
    plt.title('Neighbor ' + str(i))
    plt.subplot(1, 3, 3)
    plt.imshow(distant[RGB_BANDS, :, :])
    plt.title('Distant ' + str(i))
plt.savefig('../../RemoteSensing/exploratory_plots/example_tiles.png')

# Read mean/standard deviation for each band, for standardization purposes
train_statistics = pd.read_csv(BAND_STATISTICS_FILE)
train_means = train_statistics['mean'].values
train_stds = train_statistics['std'].values
band_means = train_means[:-1]
band_stds = train_stds[:-1]

augment = True
batch_size = 50
shuffle = True
num_workers = 4
n_triplets = 100000
bands=14

cuda = torch.cuda.is_available()
dataloader = triplet_dataloader(tile_dir, band_means, band_stds, augment=augment,
                                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                n_triplets=n_triplets, pairs_only=True)
in_channels = bands
z_dim = 20

# Set up network
TileNet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
TileNet.train()
if cuda: TileNet.cuda()


# Set up optimizer
lr = 1e-3
optimizer = optim.Adam(TileNet.parameters(), lr=lr, betas=(0.5, 0.999))
epochs = 50
margin = 10
l2 = 0.01
print_every = 10000
save_models = True
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

print('Begin training.................')
for epoch in range(0, epochs):
    print('===================== EPOCH', epoch, '========================')
    (avg_loss, avg_l_n, avg_l_d, avg_l_nd) = train_triplet_epoch(
        TileNet, cuda, dataloader, optimizer, epoch+1, margin=margin, l2=l2,
        print_every=print_every)

# Save model after last epoch
if save_models:
    model_fn = os.path.join(MODEL_DIR, 'TileNet_epoch50.ckpt')
    torch.save(TileNet.state_dict(), model_fn)
