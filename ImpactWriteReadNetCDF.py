"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature
from matplotlib.colors import LogNorm

class ImpactWriteNetCDF:
    """
    A class to write Impact data to a NetCDF file.

    Attributes:
        impact_object (Impact): The impact object containing the data to be written.
    """
    def __init__(self, impact_object):
        """Initialize the ImpactWriteNetCDF with an impact object."""
        self.eai_exp_data = impact_object.eai_exp
        self.lat_data = impact_object.coord_exp[:, 0]
        self.lon_data = impact_object.coord_exp[:, 1]
        
    def reshape_data(self):
        """Reshape the data for writing to NetCDF."""
        unique_lats, lat_inverse = np.unique(self.lat_data, return_inverse=True)
        unique_lons, lon_inverse = np.unique(self.lon_data, return_inverse=True)
        
        self.eai_exp_data_reshaped = np.full((len(unique_lats), len(unique_lons)), np.nan)
        self.eai_exp_data_reshaped[lat_inverse, lon_inverse] = self.eai_exp_data
        
        self.unique_lats = unique_lats
        self.unique_lons = unique_lons
        
    def create_dataset(self):
        """Create an xarray dataset from the reshaped data."""
        coords = {'lat': self.unique_lats, 'lon': self.unique_lons}
        data = {'eai_exp': (['lat', 'lon'], self.eai_exp_data_reshaped)}
        self.ds = xr.Dataset(data, coords=coords)
        self.ds['lat'].attrs.update({
            'standard_name': 'latitude',
            'units': 'degrees_north',
            'axis': 'Y',
            'crs': 'EPSG:4326'
        })
        self.ds['lon'].attrs.update({
            'standard_name': 'longitude',
            'units': 'degrees_east',
            'axis': 'X',
            'crs': 'EPSG:4326'
        })
        
    def save_to_netcdf(self, filename):
        """Save the dataset to a NetCDF file."""
        self.ds.to_netcdf(filename)
        
    def write_to_netcdf(self, filename):
        """Full process to write impact data to a NetCDF file."""
        self.reshape_data()
        self.create_dataset()
        self.save_to_netcdf(filename)




class ImpactReaderNetCDF:
    """
    A class to read and visualize Impact data from a NetCDF file.

    Attributes:
        filename (str): Path to the NetCDF file.
    """
    def __init__(self, filename):
        """Initialize the ImpactReaderNetCDF with a filename."""
        self.filename = filename

    def read_netcdf(self):
        """Read data from a NetCDF file."""
        self.ds = xr.open_dataset(self.filename)
        self.lat = self.ds['lat'].values
        self.lon = self.ds['lon'].values
        self.data = self.ds['eai_exp'].values  

    def visualize(self, scale='normal'):
        """
        Visualize the data.

        Args:
            scale (str): The scale to use for visualization, 'normal' or 'log'.
        """
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        
        if scale == 'log':
            norm = LogNorm()
            label = 'eai_exp (Log Scale)'
        else:
            norm = None
            label = 'eai_exp'
        
        sc = ax.pcolormesh(self.lon, self.lat, self.data, cmap='RdYlBu_r', 
                           transform=ccrs.PlateCarree(), norm=norm)
        
        ax.set_extent([self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()], crs=ccrs.PlateCarree())
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
        
        plt.colorbar(sc, ax=ax, label=label, pad=0.05)
        plt.title('Expected Annual Impact')
        plt.show()
