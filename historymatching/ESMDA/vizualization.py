#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import xarray as xr


class DataComparer:
    def __init__(self, reference_folder, pickle_filename, nc_filename, data_folder_path, monitoring_positions):
        self.reference_folder = reference_folder
        self.pickle_filename = pickle_filename
        self.nc_filename = nc_filename
        self.data_folder_path = data_folder_path
        self.monitoring_positions = monitoring_positions
      
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
        data = pd.read_pickle(filename)
        return np.array(data.values) if isinstance(data, pd.DataFrame) else data
    def compare_data(self, DPrior, dObs, DPosterior, CeDiag, time_range, num_realizations=100, y_limits=[200, 320]):
        DPrior_reshaped = DPrior.reshape((4, len(time_range), -1))[:, :, :num_realizations]
        dObs_reshaped = dObs.reshape((4, len(time_range)))
        DPosterior_reshaped = DPosterior.reshape((4, len(time_range), -1))[:, :, :num_realizations]
        CeDiag_reshaped = CeDiag.reshape((4, len(time_range)))

        fig, axes = plt.subplots(4, 2, figsize=(15, 20), sharey=True)
        for i, (ax_prior, ax_posterior) in enumerate(axes):
            # Observed Data
            ax_prior.errorbar(time_range, dObs_reshaped[i], yerr=CeDiag_reshaped[i], fmt='o', color='r', label='Observed Data')
            ax_posterior.errorbar(time_range, dObs_reshaped[i], yerr=CeDiag_reshaped[i], fmt='o', color='r', label='Observed Data')

            # Prior Individual Curves
            for realization in DPrior_reshaped[i, :, :].T:
                ax_prior.plot(time_range, realization, color='gray', alpha=0.3)

            # Posterior Individual Curves
            for realization in DPosterior_reshaped[i, :, :].T:
                ax_posterior.plot(time_range, realization, color='blue', alpha=0.3)

            # Prior Statistics
            prior_p10 = np.percentile(DPrior_reshaped[i, :, :], 10, axis=1)
            prior_p90 = np.percentile(DPrior_reshaped[i, :, :], 90, axis=1)
            prior_mean = DPrior_reshaped[i, :, :].mean(axis=1)
            ax_prior.fill_between(time_range, prior_p10, prior_p90, color='gray', alpha=0.5, label='Prior (P10-P90)')
            ax_prior.plot(time_range, prior_mean, 'k-', lw=2, label='Prior Mean')

            # Posterior Statistics
            posterior_p10 = np.percentile(DPosterior_reshaped[i, :, :], 10, axis=1)
            posterior_p50 = np.percentile(DPosterior_reshaped[i, :, :], 50, axis=1)
            posterior_p90 = np.percentile(DPosterior_reshaped[i, :, :], 90, axis=1)
            ax_posterior.fill_between(time_range, posterior_p10, posterior_p90, color='blue', alpha=0.5, label='Posterior (P10-P90)')
            ax_posterior.plot(time_range, posterior_p50, 'k-', lw=2, label='Posterior Mean')

            ax_prior.set_ylim(y_limits)
            ax_prior.set_title(f'Monitoring Point {i+1} - Prior')
            ax_posterior.set_title(f'Monitoring Point {i+1} - Posterior')
            ax_prior.set_xlabel('Time (days)')
            ax_posterior.set_xlabel('Time (days)')
            ax_prior.set_ylabel('Pressure (bar)')
            ax_prior.legend(loc='upper left')
            ax_posterior.legend(loc='upper left')


        plt.tight_layout()
        plt.savefig(os.path.join(figure_path, 'comparison_plot.jpg'), format='jpg')
        plt.show()

        

    def process_data(self, prior_filename, posterior_filename, figure_path,  num_realizations=100):
        DPrior = self.load_data(prior_filename)
        DPosterior = self.load_data(posterior_filename)
        dObs = self.get_observation_data()
        CeDiag = np.where(np.array(0.05*dObs[:])<1e-3, 1e-3, 0.01*dObs[:])
        self.compare_data(DPrior, dObs, DPosterior, CeDiag, self.time_range, figure_path , num_realizations)
       


    def plot_observation_points(self):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(self.reference_model['Perm'].values, cmap='viridis')
        for (i,j) in self.monitoring_positions:
            ax.plot(i, j, 'o', color='red', markersize=10)
        ax.set_title('Observation Points in the Permeability Field')
        plt.tight_layout()
        plt.show()