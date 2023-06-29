import numpy as np
from scipy import ndimage
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

# Function to generate random Gaussian 2D permeability field
def generate_gaussian_map(loc=50, scale=10, size=(100, 100)):
    np.random.seed(0)  # for reproducibility
    return np.random.normal(loc=loc, scale=scale, size=size)

# Function to generate non-Gaussian 2D permeability field with channels
def generate_channel_map(base=30, channel_value=70, channel_width=10, size=(100, 100)):
    map = np.full(size, base)  # base permeability
    half_size = size[0] // 2
    map[half_size-channel_width//2:half_size+channel_width//2, :] = channel_value  # create a horizontal channel
    map[:, half_size-channel_width//2:half_size+channel_width//2] = channel_value  # create a vertical channel
    return map

# Function to compute connectivity
def compute_connectivity(permeability_map, threshold):
    binary_map = permeability_map > threshold
    labeled_map, num_components = ndimage.measurements.label(binary_map)
    return binary_map, num_components

# Function to compute entropy
def compute_entropy(permeability_map, neighborhood_radius):
    # Convert permeability map to unsigned 8-bit integer type
    permeability_map_ubyte = img_as_ubyte((permeability_map - np.min(permeability_map)) / (np.max(permeability_map) - np.min(permeability_map)))
    entropy_map = entropy(permeability_map_ubyte, disk(neighborhood_radius))
    mean_entropy = np.mean(entropy_map)
    return entropy_map, mean_entropy

# Function to plot maps
def plot_maps(permeability_map, binary_map, entropy_map, title):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    axs[0].imshow(permeability_map, cmap='viridis')
    axs[0].set_title(f'{title} Permeability Map')
    axs[1].imshow(binary_map, cmap='gray')
    axs[1].set_title(f'{title} Binary Map for Connectivity')
    axs[2].imshow(entropy_map, cmap='magma')
    axs[2].set_title(f'{title} Entropy Map')
    plt.show()

# Compute and print metrics
threshold = 60  # adjust as needed
neighborhood_radius = 10  # adjust as needed

# Gaussian map
gaussian_permeability_map = generate_gaussian_map()
binary_map, num_components = compute_connectivity(gaussian_permeability_map, threshold)
entropy_map, mean_entropy = compute_entropy(gaussian_permeability_map, neighborhood_radius)
print(f'Gaussian map: number of connected components = {num_components}')
print(f'Gaussian map: mean entropy = {mean_entropy}')
plot_maps(gaussian_permeability_map, binary_map, entropy_map, 'Gaussian')

# Channel map
channel_permeability_map = generate_channel_map()
binary_map, num_components = compute_connectivity(channel_permeability_map, threshold)
entropy_map, mean_entropy = compute_entropy(channel_permeability_map, neighborhood_radius)
print(f'Channel map: number of connected components = {num_components}')
print(f'Channel map: mean entropy = {mean_entropy}')
plot_maps(channel_permeability_map, binary_map, entropy_map, 'Channel')

# Another example with more channels
channel_permeability_map = generate_channel_map(channel_value=70, channel_width=5)
binary_map, num_components = compute_connectivity(channel_permeability_map, threshold)
entropy_map, mean_entropy = compute_entropy(channel_permeability_map, neighborhood_radius)
print(f'More Channels map: number of connected components = {num_components}')
print(f'More Channels map: mean entropy = {mean_entropy}')
plot_maps(channel_permeability_map, binary_map, entropy_map, 'More Channels')


