import numpy as np
#import gdal
import os
import random

# def load_img(img_file, val_type='uint8', bands_only=False, num_bands=4):
#     """
#     Loads an image using gdal, returns it as an array.
#     """
#     obj = gdal.Open(img_file)
#     if val_type == 'uint8':
#         img = obj.ReadAsArray().astype(np.uint8)
#     elif val_type == 'float32':
#         img = obj.ReadAsArray().astype(np.float32)
#     else:
#         raise ValueError('Invalid val_type for image values. Try uint8 or float32.')
#     img = np.moveaxis(img, 0, -1)
#     if bands_only: img = img[:,:,:num_bands]
#     return img


# Compute L1-norm between anchor and neighbor crop type distributions
def crop_type_distance(anchor_tile, neighbor_tile):
    CROP_TYPE_INDICES = [9, 10, 11, 12]
    #print('========================')
    #print('anchor tile shape', anchor_tile.shape)
    anchor_crop_masks = anchor_tile[CROP_TYPE_INDICES, :, :]
    neighbor_crop_masks = neighbor_tile[CROP_TYPE_INDICES, :, :]
    anchor_crop_types = np.mean(anchor_crop_masks, axis=(1,2))
    neighbor_crop_types = np.mean(neighbor_crop_masks, axis=(1,2))
    assert(anchor_crop_types.shape[0] == 4)
    l1_norm = np.linalg.norm(anchor_crop_types - neighbor_crop_types, ord=1)
    #print('Anchor crop types', anchor_crop_types)
    #print('Neighbor crop types', neighbor_crop_types)
    #print('L1 norm', l1_norm)
    return l1_norm


# Compute fraction of pixels where data is missing
def fraction_missing_pixels(tile):
    MISSING_DATA_IDX = -1
    return np.mean(tile[MISSING_DATA_IDX, :, :])


def get_triplet_imgs(img_dir, img_ext='.tif', n_triplets=1000):
    """
    Returns a numpy array of dimension (n_triplets, 2). First column is
    the img name of anchor/neighbor tiles and second column is img name 
    of distant tiles.
    """
    img_names = []
    for filename in os.listdir(img_dir):
        if filename.endswith(img_ext):
            img_names.append(filename)
    img_triplets = list(map(lambda _: random.choice(img_names), range(2 * n_triplets)))
    img_triplets = np.array(img_triplets)
    return img_triplets.reshape((-1, 2))

def get_triplet_tiles(tile_dir, img_dir, img_triplets, tile_size=50, neighborhood=100,
                      save=True, verbose=False, MAX_CROP_TYPE_DISTANCE=0.4,
                      MAX_MISSING_PIXELS=0.5):
    # We only want to load each image into memory once. For each unique image,
    # load it into memory, and then loop through "img_triplets" to find which
    # sub-tiles should come from that image.
    if not os.path.exists(tile_dir):
        os.makedirs(tile_dir)
    size_even = (tile_size % 2 == 0)
    tile_radius = tile_size // 2

    n_triplets = img_triplets.shape[0]
    unique_imgs = np.unique(img_triplets)
    tiles = np.zeros((n_triplets, 3, 2), dtype=np.int16)

    for img_name in unique_imgs:
        print("Sampling image {}".format(img_name))
        # if img_name[-3:] == 'npy':
        #     img = np.load(img_name)
        # else:
        #     img = load_img(os.path.join(img_dir, img_name), val_type=val_type,
        #                bands_only=bands_only)
        # Pad image with 0's. I don't think this is necessary?
        # img_padded = np.pad(img, pad_width=[(tile_radius, tile_radius),
        #                                     (tile_radius, tile_radius), (0,0)],
        #                     mode='reflect')

        assert (img_name[-3:] == 'npy')
        img_padded = np.load(os.path.join(img_dir, img_name))  # TODO Reshape???
        img_shape = img_padded.shape

        for idx, row in enumerate(img_triplets):
            if row[0] == img_name:
                # From this image, sample an "anchor" and "neighbor" subtile that are close to each other 
                # keep sampling until we find anchor/neighbor with similar crop type, AND are not cloud-covered
                tries = 0
                found_good_neighbors = False
                while not found_good_neighbors:
                    xa, ya = sample_anchor(img_shape, tile_radius)
                    xn, yn = sample_neighbor(img_shape, xa, ya, neighborhood, tile_radius)
                    if verbose:
                        print("    Saving anchor and neighbor tile #{}".format(idx))
                        print("    Anchor tile center:{}".format((xa, ya)))
                        print("    Neighbor tile center:{}".format((xn, yn)))
                
                    # if save:
                    tile_anchor = extract_tile(img_padded, xa, ya, tile_radius)
                    tile_neighbor = extract_tile(img_padded, xn, yn, tile_radius)
                    if size_even:
                        tile_anchor = tile_anchor[:, :-1,:-1]
                        tile_neighbor = tile_neighbor[:, :-1,:-1]
                    tries += 1
                    print('fraction missing', fraction_missing_pixels(tile_anchor))
                    if (fraction_missing_pixels(tile_anchor) <= MAX_MISSING_PIXELS and fraction_missing_pixels(tile_neighbor) <= MAX_MISSING_PIXELS and crop_type_distance(tile_anchor, tile_neighbor) <= MAX_CROP_TYPE_DISTANCE):
                        found_good_neighbors = True
                    else:
                        print('Try again.')
                    if tries > 20:
                        print('Failed to find good tile even after 20 tries.')
                        break

                np.save(os.path.join(tile_dir, '{}anchor.npy'.format(idx)), tile_anchor)
                np.save(os.path.join(tile_dir, '{}neighbor.npy'.format(idx)), tile_neighbor)
                tiles[idx,0,:] = xa - tile_radius, ya - tile_radius
                tiles[idx,1,:] = xn - tile_radius, yn - tile_radius
                
                if row[1] == img_name:
                    # distant image is same as anchor/neighbor image
                    found_good_tile = False
                    tries = 0
                    while not found_good_tile:
                        tries += 1
                        xd, yd = sample_distant_same(img_shape, xa, ya, neighborhood, tile_radius)
                        if verbose:
                            print("    Saving distant tile #{}".format(idx))
                            print("    Distant tile center:{}".format((xd, yd)))
                        if save:
                            tile_distant = extract_tile(img_padded, xd, yd, tile_radius)
                            if size_even:
                                tile_distant = tile_distant[:, :-1,:-1]
                            if fraction_missing_pixels(tile_distant) <= MAX_MISSING_PIXELS:
                                found_good_tile = True
                        if tries > 20:
                            print('Failed to find good tile even after 20 tries.')
                            break

                    np.save(os.path.join(tile_dir, '{}distant.npy'.format(idx)), tile_distant)
                    tiles[idx,2,:] = xd - tile_radius, yd - tile_radius
            
            elif row[1] == img_name: 
                # distant image is different from anchor/neighbor image
                found_good_tile = False
                tries = 0
                while not found_good_tile:
                    xd, yd = sample_distant_diff(img_shape, tile_radius)
                    if verbose:
                        print("    Saving distant tile #{}".format(idx))
                        print("    Distant tile center:{}".format((xd, yd)))
                    if save:
                        tile_distant = extract_tile(img_padded, xd, yd, tile_radius)
                        if size_even:
                            tile_distant = tile_distant[:, :-1,:-1]
                        if fraction_missing_pixels(tile_distant) <= MAX_MISSING_PIXELS:
                            found_good_tile = True
                    if tries > 20:
                        print('Failed to find good tile even after 20 tries.')
                        break

                np.save(os.path.join(tile_dir, '{}distant.npy'.format(idx)), tile_distant)
                tiles[idx,2,:] = xd - tile_radius, yd - tile_radius
            
    return tiles

def sample_anchor(img_shape, tile_radius):
    c, w_padded, h_padded = img_shape
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius
    
    xa = np.random.randint(0, w) + tile_radius
    ya = np.random.randint(0, h) + tile_radius
    return xa, ya

def sample_neighbor(img_shape, xa, ya, neighborhood, tile_radius):
    c, w_padded, h_padded = img_shape
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius
    
    xn = np.random.randint(max(xa-neighborhood, tile_radius),
                           min(xa+neighborhood, w+tile_radius))
    yn = np.random.randint(max(ya-neighborhood, tile_radius),
                           min(ya+neighborhood, h+tile_radius))
    return xn, yn


def sample_distant_same(img_shape, xa, ya, neighborhood, tile_radius):
    c, w_padded, h_padded = img_shape
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius
    
    xd, yd = xa, ya
    while (xd >= xa - neighborhood) and (xd <= xa + neighborhood):
        xd = np.random.randint(0, w) + tile_radius
    while (yd >= ya - neighborhood) and (yd <= ya + neighborhood):
        yd = np.random.randint(0, h) + tile_radius
    return xd, yd


def sample_distant_diff(img_shape, tile_radius):
    return sample_anchor(img_shape, tile_radius)

def extract_tile(img_padded, x0, y0, tile_radius):
    """
    Extracts a tile from a (padded) image given the row and column of
    the center pixel and the tile size. E.g., if the tile
    size is 15 pixels per side, then the tile radius should be 7.
    """
    c, w_padded, h_padded = img_padded.shape
    row_min = x0 - tile_radius
    row_max = x0 + tile_radius
    col_min = y0 - tile_radius
    col_max = y0 + tile_radius
    assert row_min >= 0, 'Row min: {}'.format(row_min)
    assert row_max <= w_padded, 'Row max: {}'.format(row_max)
    assert col_min >= 0, 'Col min: {}'.format(col_min)
    assert col_max <= h_padded, 'Col max: {}'.format(col_max)
    tile = img_padded[:, row_min:row_max+1, col_min:col_max+1]
    return tile

