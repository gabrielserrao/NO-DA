#%%
import xarray as xr
import os
import numpy as np

reference_folder = '/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/REFERENCE'
#read reference metadata from reference_folder
reference = xr.open_dataset(os.path.join(reference_folder,'ReferenceSimulation.nc'))
prior = reference.copy()
GeoFolder = '/samoa/data/smrserraoseabr/NO-DA/dataset/DARTS/runnedmodels/filtered'
prior_geomodels_path ='/samoa/data/smrserraoseabr/NO-DA/historymatching/ESMDA/prior_geomodels'
Ne = 1000
treatGeoModel = False
i = 0
for model in os.listdir(GeoFolder):
    while i < Ne:
        geomodel = xr.open_dataset(os.path.join(GeoFolder, model))
        if treatGeoModel == True:
            factor = 1.01324997e12
            geomodel['Perm'] = geomodel['Perm'] * factor
            # Set the minimum value of Perm to 0.001 mD
            geomodel['Perm'] = geomodel['Perm'].where(geomodel['Perm'] >= 0.001, 0.001)

            #Set the minimum value of Por to 0.01
            geomodel['Por'] = geomodel['Por'].where(geomodel['Por'] >= 0.01, 0.01)

        prior['Perm'] = geomodel['Perm']
        prior['Por'] = geomodel['Por']
        #save prior to netcdf on the folder
        prior.to_netcdf(f'{prior_geomodels_path}/prior_{i}.nc')
        i += 1















# %%
