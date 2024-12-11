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

import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature
from matplotlib.colors import LogNorm
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=mpl.MatplotlibDeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")


class ImpactReaderNetCDF:
    """
    A class designed to read and visualize impact data from NetCDF files using geographical and cartographic representations.

    Attributes:
        filename (str): Path to the NetCDF file containing impact data.
    """

    def __init__(self, filename):
        """
        Initializes the ImpactReaderNetCDF object with the file path to a NetCDF dataset.

        Parameters:
            filename (str): The full path to the NetCDF file to be read.
        """
        self.filename = filename
        self.ds = None  # Will hold the xarray dataset after reading the NetCDF file
        self.lat = None  # Will hold the latitude coordinates
        self.lon = None  # Will hold the longitude coordinates

    def read_netcdf(self):
        """
        Reads the specified NetCDF file, dynamically identifying the latitude and longitude coordinates.
        Raises a KeyError if these coordinates cannot be found within the dataset.
        """
        self.ds = xr.open_dataset(self.filename)
        # Attempt to find latitude and longitude coordinates with common naming conventions
        lat_names = ['lat', 'latitude']
        lon_names = ['lon', 'longitude']
        for name in lat_names:
            if name in self.ds.coords:
                self.lat = self.ds[name].values
                break
        for name in lon_names:
            if name in self.ds.coords:
                self.lon = self.ds[name].values
                break
        if self.lat is None or self.lon is None:
            raise KeyError("Latitude or longitude coordinate not found in dataset.")

    def visualize_eai_exp(self, scale='normal'):
        """
        Visualizes the Expected Annual Impact Exposure (eai_exp) data from the NetCDF file.

        Parameters:
            scale (str): Determines the scaling of the visualization ('normal' for linear, 'log' for logarithmic).
        """
        data = self.ds['eai_exp'].values
        self._visualize(data, 'eai_exp', scale)

    def visualize_imp_mat(self, scale='normal'):
        """
        Visualizes the Impact Matrix (imp_mat) data from the NetCDF file. If the data includes a time dimension,
        it averages over time for visualization purposes.

        Parameters:
            scale (str): Determines the scaling of the visualization ('normal' for linear, 'log' for logarithmic).
        """
        if 'time' in self.ds['imp_mat'].dims:
            data = self.ds['imp_mat'].isel(time=0).values
        else:
            data = self.ds['imp_mat'].values
        self._visualize(data, 'imp_mat', scale)

    def _visualize(self, data, title, scale):
        """
        A helper method to create visualizations of the impact data, supporting both linear and logarithmic scales.

        Parameters:
            data (numpy.ndarray): The data array to be visualized.
            title (str): The title for the plot, indicating the type of data being visualized.
            scale (str): The scale for the visualization ('normal' or 'log').
        """
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        norm = LogNorm() if scale == 'log' else None
        label = f'{title} ({"Log Scale" if scale == "log" else "Linear Scale"})'
        sc = ax.pcolormesh(self.lon, self.lat, data, cmap='RdYlBu_r', transform=ccrs.PlateCarree(), norm=norm)
        ax.set_extent([self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max()], crs=ccrs.PlateCarree())
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
        plt.colorbar(sc, ax=ax, label=label, pad=0.05)
        plt.title(f'{title} Visualization')
        plt.show()

# Example usage 
# Please replace `filename` with the actual path to your NetCDF file containing impact data
#filename = 'path/to/your/impact_data.nc'
#reader = ImpactReaderNetCDF(filename)
#reader.read_netcdf()
#reader.visualize_eai_exp(scale='log')  # Visualize eai_exp with logarithmic scale
#reader.visualize_imp_mat(scale='log')  # Visualize imp_mat with logarithmic scale