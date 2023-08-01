#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import xarray as xr


class DataComparer:
    def __init__(self, reference_folder, pickle_filename, nc_filename, data_folder_path):
        self.reference_folder = reference_folder
        self.pickle_filename = pickle_filename
        self.nc_filename = nc_filename
        self.data_folder_path = data_folder_path
        self.monitoring_positions = [[11, 11], [11, 21], [21, 11], [21, 21]]
 #wells position in the 20x20 grid
        self.load_metadata()

    def load_metadata(self):
        with open(os.path.join(self.reference_folder, self.pickle_filename), 'rb') as f:
            metadata = pickle.load(f)

        self.dt = metadata.iloc[0].values[0]
        self.nsteps = metadata.iloc[1].values[0]
        self.well_coords = metadata.iloc[2].values[0]
        self.well_rates = metadata.iloc[3].values[0]
        self.initial_gas_rate = metadata.iloc[4].values[0]
        self.treatGeoModel = metadata.iloc[7].values[0]
        self.RefGeoData_path = metadata.iloc[8].values[0]
        self.reference_model = xr.open_dataset(os.path.join(self.reference_folder, self.nc_filename))
        self.time_range = self.reference_model.time.values

    def get_observation_data(self):
        obsData = []
        for (i,j) in self.monitoring_positions:
            obsData.append(self.reference_model['Pressure'].isel(X=i).isel(Y=j).values)
        return np.array(obsData).flatten()

    def load_data(self, filename):
        data = pd.read_pickle(os.path.join(self.data_folder_path, filename))
        return np.array(data.values) if isinstance(data, pd.DataFrame) else data

    def compare_data(self, DPrior, dObs, DPosterior, CeDiag, time_range, num_realizations=100, y_limits=[200,320]):
        DPrior_reshaped = DPrior.reshape((4, len(time_range), -1))[:, :, :num_realizations]
        dObs_reshaped = dObs.reshape((4, len(time_range)))
        DPosterior_reshaped = DPosterior.reshape((4, len(time_range), -1))[:, :, :num_realizations]
        CeDiag_reshaped = CeDiag.reshape((4, len(time_range)))

        fig, axes = plt.subplots(4, 1, figsize=(10, 20))
        for i, ax in enumerate(axes):
            ax.errorbar(time_range, dObs_reshaped[i], yerr=CeDiag_reshaped[i], fmt='o', color='r', label='Observed Data')
            
            # Plot Prior as a shaded region
            prior_min = DPrior_reshaped[i, :, :].min(axis=1)
            prior_max = DPrior_reshaped[i, :, :].max(axis=1)
            ax.fill_between(time_range, prior_min, prior_max, color='gray', alpha=0.2, label='DPrior')
            
            # Calculate percentiles for Posterior
            posterior_p10 = np.percentile(DPosterior_reshaped[i, :, :], 10, axis=1)
            posterior_p50 = np.percentile(DPosterior_reshaped[i, :, :], 50, axis=1)
            posterior_p90 = np.percentile(DPosterior_reshaped[i, :, :], 90, axis=1)
            
            # Plot Posterior percentiles
            ax.plot(time_range, posterior_p10, 'blue', alpha=0.5, label='DPosterior P10', linestyle='--')
            ax.plot(time_range, posterior_p50, 'blue', alpha=0.8, label='DPosterior P50')
            ax.plot(time_range, posterior_p90, 'blue', alpha=0.5, label='DPosterior P90', linestyle='--')

            ax.set_ylim(y_limits)
            ax.set_title(f'Observation Data vs DPrior vs DPosterior for Monitoring Point {i+1}')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Pressure (bar)')
            ax.legend()
        
        plt.tight_layout()
        plt.show()
       

    def process_data(self, prior_filename, posterior_filename, num_realizations=100):
        DPrior = self.load_data(prior_filename)
        DPosterior = self.load_data(posterior_filename)
        dObs = self.get_observation_data()
        CeDiag = np.where(np.array(0.05*dObs[:])<1e-3, 1e-3, 0.01*dObs[:])
        self.compare_data(DPrior, dObs, DPosterior, CeDiag, self.time_range, num_realizations)


    def plot_observation_points(self):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(self.reference_model['Perm'].values, cmap='viridis')
    for (i,j) in self.monitoring_positions:
        ax.plot(i, j, 'o', color='red', markersize=10)
    ax.set_title('Observation Points in the Permeability Field')
    plt.tight_layout()
    plt.show()

#%%
if __name__ == "__main__":
    comparer = DataComparer(reference_folder='/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/REFERENCE',
                            pickle_filename='Reference_metadata.pkl',
                            nc_filename='ReferenceSimulation.nc',
                            data_folder_path='/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA')

    
    comparer.process_data(prior_filename='simulations_HF/it0/D_0.pkl', posterior_filename='simulations_HF/it3/D_3.pkl', title='HF')
    comparer.process_data(prior_filename='simulations_Px/it0/D_0.pkl', posterior_filename='simulations_Px/it3/D_3.pkl', title='Px')
    comparer.process_data(prior_filename='simulations_HF10/it0/D_0.pkl', posterior_filename='simulations_HF10/it3/D_3.pkl', title='HF10')
    comparer.process_data(prior_filename='simulations_HF50_0/it0/D_0.pkl', posterior_filename='simulations_HF50_0/it3/D_3.pkl', title='HF50_0')
    comparer.process_data(prior_filename='simulations_HF50_500/it0/D_0.pkl', posterior_filename='simulations_HF50_500/it3/D_3.pkl', title='HF50_500')


# %%

# %%
