#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import imageio
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import re
from darts.engines import value_vector
#load the data realizations.nc file into an xarray dataset called realizations

def create_wells_dataset(well_names, well_rates, well_types, well_coords, nsteps, initial_gas_rate):
    """Create an xarray Dataset of well data.
    
    Parameters:
    well_names (list of str): Names of the wells.
    well_types (list of str): Types of the wells.
    well_coords (list of tuple): Coordinates of the wells.
    nsteps (int): Number of time steps.
    initial_gas_rate (float): Initial gas rate.
    
    Returns:
    xarray.Dataset: Dataset of well data.
    """
  
    # Create dictionary for Dataset
    data_dict = {
        'WellType': (('WellName'), well_types),
        'i': (('WellName'), [coord[0] for coord in well_coords]),
        'j': (('WellName'), [coord[1] for coord in well_coords]),
    }

    # Create Dataset
    wells_config = xr.Dataset(
        data_dict,
        coords={'WellName': well_names, 'steps': np.arange(nsteps)}
    )

    # Add gas rates to Dataset
    wells_config['gas_rate'] = (('WellName', 'steps'), np.outer(np.ones(len(well_coords)), well_rates))

    return wells_config

def InitializeDataVars(realization, nsteps, times):
    data_vars = {
        'GasSat': (['time','Y', 'X'], np.zeros((nsteps+1, realization.dims['Y'], realization.dims['X']))),
        'WatSat': (['time','Y', 'X'], np.zeros((nsteps+1, realization.dims['Y'], realization.dims['X']))),
        'Pressure': (['time','Y', 'X'], np.zeros((nsteps+1, realization.dims['Y'], realization.dims['X']))),
        'CO_2': (['time','Y', 'X'], np.zeros((nsteps+1, realization.dims['Y'], realization.dims['X']))),
        'C_1': (['time','Y', 'X'], np.zeros((nsteps+1, realization.dims['Y'], realization.dims['X']))),
        'H2O': (['time','Y', 'X'], np.zeros((nsteps+1, realization.dims['Y'], realization.dims['X']))),
        'Perm': (['Y', 'X'], np.zeros((realization.dims['Y'], realization.dims['X']))),
        'Por': (['Y', 'X'], np.zeros((realization.dims['Y'], realization.dims['X']))),
    
    }

    attrs = {'title': 'Simulation Results'}
    data = xr.Dataset(data_vars=data_vars, coords={'time': times, 'X': realization['X'], 'Y': realization['Y']}, attrs=attrs)
    data['GasSat'] = xr.DataArray(np.zeros((nsteps+1, realization.dims['Y'], realization.dims['X'])), dims=['time','Y', 'X'])
    data['WatSat'] = xr.DataArray(np.zeros((nsteps+1, realization.dims['Y'], realization.dims['X'])), dims=['time','Y', 'X'])
    data['Pressure'] = xr.DataArray(np.zeros((nsteps+1, realization.dims['Y'], realization.dims['X'])), dims=['time','Y', 'X'])
    data['CO_2'] = xr.DataArray(np.zeros((nsteps+1, realization.dims['Y'], realization.dims['X'])), dims=['time','Y', 'X'])
    data['C_1'] = xr.DataArray(np.zeros((nsteps+1, realization.dims['Y'], realization.dims['X'])), dims=['time','Y', 'X'])
    data['H2O'] = xr.DataArray(np.zeros((nsteps+1, realization.dims['Y'], realization.dims['X'])), dims=['time','Y', 'X'])
    data['Perm'] = xr.DataArray(np.zeros((realization.dims['Y'], realization.dims['X'])), dims=['Y', 'X'])
    data['Por'] = xr.DataArray(np.zeros((realization.dims['Y'], realization.dims['X'])), dims=['Y', 'X'])

    return data


def StoreSimValues(m, size, data, t):
     
    P = np.array(m.physics.engine.X[0:m.reservoir.nb*3:3]).reshape((size, size))
    z1 = np.array(m.physics.engine.X[1:m.reservoir.nb*3:3]).reshape((size, size))
    z2 = np.array(m.physics.engine.X[2:m.reservoir.nb*3:3]).reshape((size, size))
    z3 = 1 - z1 - z2
    sg = np.zeros((size, size))
    sw = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            values = value_vector([0] * m.physics.n_ops)
            state = value_vector((P[i, j], z1[i, j], z2[i, j]))
            m.physics.property_itor.evaluate(state, values)
            sg[i, j] = values[0]
            sw[i, j] = 1 - sg[i, j]
    data['GasSat'][t] = xr.DataArray(sg, dims=('Y', 'X'), coords={'X': data['X'], 'Y': data['Y']})
    data['WatSat'][t] = xr.DataArray(sw, dims=('Y', 'X'), coords={'X': data['X'], 'Y': data['Y']})
    data['Pressure'][t] = xr.DataArray(P, dims=('Y', 'X'), coords={'X': data['X'], 'Y': data['Y']})
    data['CO_2'][t] = xr.DataArray(z1, dims=('Y', 'X'), coords={'X': data['X'], 'Y': data['Y']})
    data['C_1'][t] = xr.DataArray(z2, dims=('Y', 'X'), coords={'X': data['X'], 'Y': data['Y']})
    data['H2O'][t] = xr.DataArray(z3, dims=('Y', 'X'), coords={'X': data['X'], 'Y': data['Y']})
    data['Perm'][:] = xr.DataArray(m.permx.reshape((size, size)), dims=('Y', 'X'), coords={'X': data['X'], 'Y': data['Y']})
    data['Por'][:] = xr.DataArray(m.poro.reshape((size, size)), dims=('Y', 'X'), coords={'X': data['X'], 'Y': data['Y']})
 
    return data




def ModelOut(m):
    re_time_data = re.compile('(?P<origin>\w*?)[\s:]*(?P<name>[\w\s]+) \(?(?P<unit>[\w\/]+)\)?')

    time = np.array(m.physics.engine.time_data['time'])
    data_arrays = []
    origins = set()
    ds = xr.Dataset()

    for k, v in m.physics.engine.time_data.items():
        if re_time_data.match(k):
            origin, name, unit = re_time_data.match(k).groups()
            #substitute spaces with underscores in all names
            name = name.replace(' ', '_')
            origin = origin.replace(' ', '_')
            
            ds = ds.merge({name:
                        xr.DataArray(
                            data=np.array(m.physics.engine.time_data[k]).reshape(1, -1) if origin else np.array(m.physics.engine.time_data[k]), 
                            coords={'origin': [origin], 'time': time} if origin else {'time': time},
                            dims=('origin', 'time') if origin else ('time'),
                            attrs={'unit': unit}
                        )
            })
    return ds

# %%

def create_gif_from_dataset(dataset, realization):
    def plot_frame(dataset, z_index):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Porosity
        dataset.Porosity.sel(Realization=realization).isel(Z=z_index).plot(ax=axes[0], cmap='viridis', add_colorbar=True)
        axes[0].set_title('Porosity')

        # Permeability
        dataset.Permeability.sel(Realization=realization).isel(Z=z_index).plot(ax=axes[1], cmap='inferno', add_colorbar=True)
        axes[1].set_title('Permeability')

        # Facies
        dataset.Facies.sel(Realization=realization).isel(Z=z_index).plot(ax=axes[2], cmap='tab10', add_colorbar=True)
        axes[2].set_title('Facies')

        plt.tight_layout()
        return fig

    frames = []
    z_values = dataset.Z.values

    for z_index in range(len(z_values)):
        fig = plot_frame(dataset, z_index)
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        frames.append(frame)

    imageio.mimsave(f'properties_animation_realization_{realization}.gif', frames, 'GIF', duration=0.5)

# Example usage:
#dataset = xr.open_dataset('geomodels/models_realization_0.nc')
#create_gif_from_dataset(dataset, realization=1)

# %%


def print_statistics(dataset):
    properties = ['Facies', 'Porosity', 'Permeability']
    
    for prop in properties:
        print(f"Statistics for {prop}:")
        data = dataset[prop].values.flatten()
        
        mean = data.mean()
        std_dev = data.std()
        min_val = data.min()
        max_val = data.max()
        q25, q75 = np.percentile(data, [25, 75])
        
        print(f"  Mean: {mean:.4f}")
        print(f"  Standard Deviation: {std_dev:.4f}")
        print(f"  Min: {min_val:.4f}")
        print(f"  Max: {max_val:.4f}")
        print(f"  25th Percentile: {q25:.4f}")
        print(f"  75th Percentile: {q75:.4f}\n")

    print("Global Attributes:")
    for attr in dataset.attrs:
        print(f"  {attr}: {dataset.attrs[attr]}")
    
    print("\nAdditional Statistics:")
    for var in dataset.variables:
        if var not in dataset.dims and var not in properties:
            print(f"  {var}: {dataset[var].values}")

# Example usage
#dataset = xr.open_dataset('geomodels/models_realization_0.nc')
#print_statistics(dataset)

# %%
def plot_fancy_graphs(dataset, realization, sample_fraction=0.1, n_bins=6):
    # Extract Porosity, Permeability, and Facies data for the given realization
    porosity_data = dataset.Porosity.sel(Realization=realization).values.flatten()
    permeability_data = dataset.Permeability.sel(Realization=realization).values.flatten()
    facies_data = dataset.Facies.sel(Realization=realization).values.flatten()

    # Bin the facies values into fewer categories
    facies_binned = pd.cut(facies_data, bins=n_bins, labels=False)

    # Create a Pandas DataFrame
    df = pd.DataFrame({'Porosity': porosity_data, 'Permeability': permeability_data, 'Facies': facies_binned})

    # Downsample the data
    df_sampled = df.sample(frac=sample_fraction)

    # Pair plot
    sns.pairplot(df_sampled, vars=['Porosity', 'Permeability'], hue='Facies', diag_kind='kde', markers='+', palette='tab10')
    plt.suptitle(f'Pair Plot of Porosity and Permeability for Realization {realization}', y=1.02)
    plt.show()

    # Box plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df, x='Porosity', ax=axes[0], color='blue')
    axes[0].set_title(f'Porosity Box Plot for Realization {realization}')
    sns.boxplot(data=df, x='Permeability', ax=axes[1], color='green')
    axes[1].set_title(f'Permeability Box Plot for Realization {realization}')
    plt.tight_layout()
    plt.show()


# Example usage
#dataset = xr.open_dataset('geomodels/models_realization_0.nc')
#plot_fancy_graphs(dataset, realization=1, sample_fraction=0.1)

# %%
