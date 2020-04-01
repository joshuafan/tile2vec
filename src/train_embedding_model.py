import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torch.optim as optim

from datasets import triplet_dataloader
from sample_tiles import get_triplet_imgs, get_triplet_tiles
from tilenet import make_tilenet
from training import train_triplet_epoch

START_DATE = "2018-08-01"
DATASET_DIR = "../../RemoteSensing/datasets/dataset_" + START_DATE
IMAGE_DIR = "../../RemoteSensing/datasets/images_" + START_DATE
BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
TILE2VEC_TILE_DATASET_DIR = "../../RemoteSensing/datasets/tile2vec_tiles_2018-08-01_neighborhood500"
MODEL_DIR = "../../RemoteSensing/models/tile2vec_dim10_neighborhood500"

if not os.path.exists(TILE2VEC_TILE_DATASET_DIR):
    os.makedirs(TILE2VEC_TILE_DATASET_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

RGB_BANDS = [1, 2, 3]
NUM_TRIPLETS = 100000

# Choose which files to sample triplets of tiles from 
img_triplets = get_triplet_imgs(IMAGE_DIR, img_ext='.npy', n_triplets=NUM_TRIPLETS)

# Get tiles
tiles = get_triplet_tiles(TILE2VEC_TILE_DATASET_DIR,
                          IMAGE_DIR,
                          img_triplets,
                          tile_size=10,
                          neighborhood=500,
                          save=True,
                          verbose=True)

# Visualize
#tile_dir = '../data/example_tiles/'
#triplets_to_visualize = 10
#plt.rcParams['figure.figsize'] = (12, 4)
#for i in range(triplets_to_visualize):
#    tile = np.load(os.path.join(TILE2VEC_TILE_DATASET_DIR, str(i) + 'anchor.npy'))
#    neighbor = np.load(os.path.join(TILE2VEC_TILE_DATASET_DIR, str(i) + 'neighbor.npy'))
#    distant = np.load(os.path.join(TILE2VEC_TILE_DATASET_DIR, str(i) + 'distant.npy'))
#    print('tile shape', tile.shape, 'dtype', tile.dtype)
#    visualize_image = np.moveaxis(neighbor[RGB_BANDS, :, :] / 1000., 0, -1)
#    print('visualized', visualize_image.shape, 'dtype', tile.dtype)
#    #vmin = np.array([tile, neighbor, distant]).min()
#    #vmax = np.array([tile, neighbor, distant]).max()

#    plt.figure()
#    plt.subplot(1, 3, 1)
#    plt.imshow(np.moveaxis(tile[RGB_BANDS, :, :] / 1000., 0, -1))
#    plt.title('Anchor ' + str(i))
#    plt.subplot(1, 3, 2)
#    plt.imshow(np.moveaxis(neighbor[RGB_BANDS, :, :] / 1000., 0, -1))
#    plt.title('Neighbor ' + str(i))
#    plt.subplot(1, 3, 3)
#    plt.imshow(np.moveaxis(distant[RGB_BANDS, :, :] / 1000., 0, -1))
#    plt.title('Distant ' + str(i))
#    plt.savefig('../../RemoteSensing/exploratory_plots/example_tiles_' + str(i) +'.png')

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
bands=14

cuda = torch.cuda.is_available() and 'CUDA_VISIBLE_DEVICES' in os.environ
print('Cuda:', cuda)
dataloader = triplet_dataloader(TILE2VEC_TILE_DATASET_DIR, band_means, band_stds, augment=augment,
                                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                n_triplets=NUM_TRIPLETS, pairs_only=True)
in_channels = bands
z_dim = 10

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
print_every = 100000
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
