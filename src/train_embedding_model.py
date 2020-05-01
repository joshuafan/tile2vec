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
import sys
sys.path.append('../../RemoteSensing')
import small_resnet

START_DATE = "2018-07-17"
DATA_DIR = "/mnt/beegfs/bulk/mirror/jyf6/datasets"
DATASET_DIR = os.path.join(DATA_DIR, "dataset_" + START_DATE)
IMAGE_DIR = os.path.join(DATA_DIR, "images_" + START_DATE)
TILE_METADATA_FILE = os.path.join(DATASET_DIR, "tile_info_train.csv")
TILE_METADATA_FILE_VAL = os.path.join(DATASET_DIR, "tile_info_val.csv")

BAND_STATISTICS_FILE = os.path.join(DATASET_DIR, "band_statistics_train.csv")
TILE2VEC_TILE_DATASET_DIR = os.path.join(DATA_DIR, "tile2vec_tiles_" + START_DATE + "_neighborhood100")
TILE2VEC_TILE_DATASET_DIR_VAL = os.path.join(DATA_DIR, "tile2vec_tiles_" + START_DATE + "_neighborhood100_val")

MODEL_DIR = os.path.join(DATA_DIR, "models/tile2vec_recon_no_bn")

if not os.path.exists(TILE2VEC_TILE_DATASET_DIR):
    os.makedirs(TILE2VEC_TILE_DATASET_DIR)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

RGB_BANDS = [1, 2, 3]
NUM_TRIPLETS = 100000
NUM_TRIPLETS_VAL = 20000
NEIGHBORHOOD = 10
CROP_TYPE_INDICES = range(12, 42)
TILE_SIZE = 10
augment = True
batch_size = 50
shuffle = True
num_workers = 4
bands=43
z_dim=256
epochs = 50
margin = 10
l2 = 1e-4
lr = 1e-3
FROM_PRETRAINED = False  #True  #False  # False

# Choose which files to sample triplets of tiles from 
#img_triplets = get_triplet_imgs(TILE_METADATA_FILE, n_triplets=NUM_TRIPLETS)  # IMAGE_DIR, img_ext='.npy', n_triplets=NUM_TRIPLETS)

# Get tiles
#tiles = get_triplet_tiles(TILE2VEC_TILE_DATASET_DIR,
                          # IMAGE_DIR,
#                          img_triplets,
#                          CROP_TYPE_INDICES,
#                          tile_size=TILE_SIZE,
#                          neighborhood=NEIGHBORHOOD,
#                          save=True,
#                          verbose=True)

# Choose which files to sample triplets of tiles from 
#img_triplets_val = get_triplet_imgs(TILE_METADATA_FILE_VAL, n_triplets=NUM_TRIPLETS_VAL)  # IMAGE_DIR, img_ext='.npy', n_triplets=NUM_TRIPLETS)

# Get tiles
#tiles_val = get_triplet_tiles(TILE2VEC_TILE_DATASET_DIR_VAL,
                           # IMAGE_DIR,
#                          img_triplets_val,
#                          CROP_TYPE_INDICES,
#                          tile_size=TILE_SIZE,
#                          neighborhood=NEIGHBORHOOD,
#                          save=True,
#                          verbose=True)


# Visualize
#tile_dir = '../data/example_tiles/'
#triplets_to_visualize = 20
#plt.rcParams['figure.figsize'] = (20, 4)
#for i in range(triplets_to_visualize):
#    tile = np.load(os.path.join(TILE2VEC_TILE_DATASET_DIR, str(i) + 'anchor.npy'))
#    neighbor = np.load(os.path.join(TILE2VEC_TILE_DATASET_DIR, str(i) + 'neighbor.npy'))
#    distant = np.load(os.path.join(TILE2VEC_TILE_DATASET_DIR, str(i) + 'distant.npy'))
#    print('tile shape', tile.shape, 'dtype', tile.dtype)
#    visualize_image = np.moveaxis(neighbor[RGB_BANDS, :, :] / 1000., 0, -1)
#    print('visualized', visualize_image.shape, 'dtype', tile.dtype)
    #vmin = np.array([tile, neighbor, distant]).min()
    #vmax = np.array([tile, neighbor, distant]).max()

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

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"
print("Device", device)
cuda = torch.cuda.is_available() and 'CUDA_VISIBLE_DEVICES' in os.environ
print('Cuda:', cuda)

dataloader = triplet_dataloader(TILE2VEC_TILE_DATASET_DIR, band_means, band_stds, augment=augment,
                                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                n_triplets=NUM_TRIPLETS, pairs_only=True)

dataloader_val = triplet_dataloader(TILE2VEC_TILE_DATASET_DIR_VAL, band_means, band_stds, augment=augment,
                                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                n_triplets=NUM_TRIPLETS_VAL, pairs_only=True)
in_channels = bands

# Set up network
TileNet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
# TileNet = small_resnet.resnet18(input_channels=in_channels, output_dim=z_dim)
if FROM_PRETRAINED:
    TileNet.load_state_dict(torch.load(os.path.join(MODEL_DIR, 'TileNet.ckpt'), map_location=device))
TileNet.train()
if cuda: TileNet.cuda()


# Set up optimizer
optimizer = optim.Adam(TileNet.parameters(), lr=lr)  # , betas=(0.5, 0.999))
print_every = 10000
save_models = True
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

print('Begin training.................')
for epoch in range(0, epochs):
    print('===================== EPOCH', epoch, '========================')
    (avg_loss, avg_l_n, avg_l_d, avg_l_nd) = train_triplet_epoch(
        TileNet, cuda, dataloader, optimizer, epoch+1, is_train=True, margin=margin, l2=l2,
        print_every=print_every)

    print('******************* validation set losses ****************')
    (avg_loss_val, avg_l_n_val, avg_l_d_val, avg_l_nd_val) = train_triplet_epoch(
        TileNet, cuda, dataloader_val, optimizer, epoch+1, is_train=False, margin=margin, l2=l2,
        print_every=print_every)


    # Save model after last epoch
    if save_models:
        model_fn = os.path.join(MODEL_DIR, 'TileNet_2.ckpt')
        torch.save(TileNet.state_dict(), model_fn)
