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


import numpy as np
import xarray as xr
import os
from scipy.sparse import csr_matrix
from datetime import datetime

def parse_date(date_str):
    """Try to parse a date string with multiple formats."""
    for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d'):
        try:
            return datetime.strptime(date_str, fmt).strftime('%d-%m-%Y')
        except ValueError:
            pass
    raise ValueError(f"Date format for '{date_str}' is not recognized")

def save_impact_data_to_NetCDF(impact_object, filename, include_eai_exp=True, include_imp_mat=True, log_scale_imp=False, log_scale_eai=False, time_attribute='event_name'):
    """
    Saves environmental and/or impact matrix data from an impact object to a NetCDF file, optionally applying logarithmic scaling to enhance data dynamics. The function provides spatial referencing and allows the selection of specific data components to be included in the output file.

    Parameters:
        impact_object : object
            The impact object containing environmental (eai_exp) and impact matrix (imp_mat) data.
        filename : str
            The path and name of the file where the NetCDF data will be saved.
        include_eai_exp : bool, optional
            If True, includes environmental data (eai_exp) in the output file. Default is True.
        include_imp_mat : bool, optional
            If True, includes the impact matrix (imp_mat) data in the output file. Default is True.
        log_scale_imp : bool, optional
            If True, applies logarithmic scaling to the impact matrix data. Default is False.
        log_scale_eai : bool, optional
            If True, applies logarithmic scaling to the environmental data. Default is False.
        time_attribute : str, optional
            The attribute name within the impact_object that contains time data, which will be used to label the time dimension in the NetCDF file. Default is 'event_name'.

    Outputs:
        Creates a NetCDF file at the specified location (`filename`). The file includes the specified data elements, potentially transformed by logarithmic scaling, and is spatially referenced.

    Raises:
        ValueError: If the specified `time_attribute` is not found in the `impact_object`.

    Example:
        save_impact_data_to_NetCDF(impact_data, 'output.nc', include_eai_exp=True, include_imp_mat=True, log_scale_imp=False, log_scale_eai=False, time_attribute='event_date')

    """
    coords = {}
    data_vars = {}

    # Validate presence of time attribute
    if not hasattr(impact_object, time_attribute) or getattr(impact_object, time_attribute) is None:
        raise ValueError(f"Attribute '{time_attribute}' not found in impact_object or is None")

    time_data = getattr(impact_object, time_attribute)
    if not time_data:
        raise TypeError(f"Time data in '{time_attribute}' is empty or None")

    # Process and potentially transform eai_exp data
    if include_eai_exp and hasattr(impact_object, 'eai_exp'):
        unique_lats, lat_inverse = np.unique(impact_object.coord_exp[:, 0], return_inverse=True)
        unique_lons, lon_inverse = np.unique(impact_object.coord_exp[:, 1], return_inverse=True)
        eai_exp_reshaped = np.full((len(unique_lats), len(unique_lons)), np.nan)
        for idx, value in enumerate(impact_object.eai_exp):
            transformed_value = np.log(value + 1) if log_scale_eai else value
            eai_exp_reshaped[lat_inverse[idx], lon_inverse[idx]] = transformed_value
        coords.update({'latitude': ('latitude', unique_lats, {'units': 'degrees_north'}),
                       'longitude': ('longitude', unique_lons, {'units': 'degrees_east'})})
        data_vars.update({'eai_exp': (('latitude', 'longitude'), eai_exp_reshaped)})

    # Process and potentially transform imp_mat data
    if include_imp_mat and hasattr(impact_object, 'imp_mat'):
        dense_imp_mat = impact_object.imp_mat.toarray() if isinstance(impact_object.imp_mat, csr_matrix) else impact_object.imp_mat
        num_time_steps, num_spatial_points = dense_imp_mat.shape
        
        # Check if unique_lats and unique_lons already exist, if not, create them
        if 'latitude' not in coords:
            unique_lats, lat_inverse = np.unique(impact_object.coord_exp[:, 0], return_inverse=True)
        if 'longitude' not in coords:
            unique_lons, lon_inverse = np.unique(impact_object.coord_exp[:, 1], return_inverse=True)
            
        imp_mat_3d = np.full((num_time_steps, len(unique_lats), len(unique_lons)), np.nan)
        for time_step in range(num_time_steps):
            for spatial_point in range(num_spatial_points):
                value = dense_imp_mat[time_step, spatial_point]
                transformed_value = np.log(value + 1) if log_scale_imp else value
                lat_idx, lon_idx = lat_inverse[spatial_point], lon_inverse[spatial_point]
                imp_mat_3d[time_step, lat_idx, lon_idx] = transformed_value

        if isinstance(time_data[0], str):
            # Adjusted to handle multiple date formats
            time_data = [parse_date(date) for date in time_data]
        coords.update({'time': ('time', time_data)})
        data_vars.update({'imp_mat': (('time', 'latitude', 'longitude'), imp_mat_3d)})

    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs['crs'] = 'EPSG:4326'
    
    # Set encoding and _FillValue only if 'imp_mat' is included
    if 'imp_mat' in data_vars:
        ds['imp_mat'].encoding['_FillValue'] = -9999
        ds['imp_mat'] = ds['imp_mat'].astype('float32')

    compression_opts = dict(zlib=True, complevel=5)
    encoding = {var: compression_opts for var in ds.data_vars}
    ds.to_netcdf(filename, mode='w', encoding=encoding)

    # Print the output path when finished
    absolute_path = os.path.abspath(filename)
    print(f"Data saved to NetCDF file at {absolute_path}")


# Example usage:
#save_impact_data_to_NetCDF(tn_impact_p95, 'tn_impact_p95_imp_mat_eai_exp_p.nc', include_eai_exp=True, include_imp_mat=True, log_scale_imp=True, log_scale_eai=True, time_attribute='event_name')



import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import LogNorm
import cartopy.feature
import warnings

# Suppress specific runtime warnings from shapely
warnings.filterwarnings('ignore', 'invalid value encountered in intersects', RuntimeWarning)


class ImpactReaderNetCDF:
    """
    A class to read and visualize either environmental or impact matrix data from a NetCDF file.

    Attributes:
        filename (str): Path to the NetCDF file.
    """

    def __init__(self, filename):
        """
        Initialize the ImpactReaderNetCDF with a filename.
        """
        self.filename = filename
        self.ds = None

    def read_netcdf(self):
        """
        Read data from the NetCDF file.
        """
        self.ds = xr.open_dataset(self.filename)

    def visualize(self, data_type='eai_exp', time_step=0, scale='linear'):
        """
        Visualize the data for a specific type and time step, including the date in the title for 'imp_mat'.

        Args:
            data_type (str): The type of data to visualize, 'eai_exp' or 'imp_mat'.
            time_step (int): The time step to visualize for 'imp_mat', including its date in the title.
            scale (str): The scale to use for visualization, 'linear' or 'log'.
        """
        if data_type not in ['eai_exp', 'imp_mat']:
            raise ValueError("data_type must be 'eai_exp' or 'imp_mat'")

        data = self.ds[data_type].isel(time=time_step) if data_type == 'imp_mat' else self.ds[data_type]
        lat = self.ds['latitude'].values
        lon = self.ds['longitude'].values

        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

        if scale == 'log':
            norm = LogNorm()  # Avoid log(0) issues by ensuring min is > 0
            label = f'{data_type} (Log Scale)'
        else:
            norm = None
            label = f'{data_type}'

        sc = ax.pcolormesh(lon, lat, data.values, cmap='RdYlBu_r', 
                           transform=ccrs.PlateCarree(), norm=norm)

        ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        ax.add_feature(cartopy.feature.BORDERS, linestyle=':')

        plt.colorbar(sc, ax=ax, label=label, pad=0.05)
        
        # Set title based on data type and conditional inclusion of time step and date for 'imp_mat'
        if data_type == 'imp_mat':
            date = str(self.ds['time'].values[time_step])[:10]  # Extracting date string in YYYY-MM-DD format
            title = f'Visualization of {data_type} - Time Step {time_step} ({date})'
        else:
            title = f'Visualization of {data_type}'

        plt.title(title)
        plt.show()

# Example usage:
# reader = ImpactReaderNetCDF('tn_impact_p95_imp_mat_eai_exp_zu.nc')
# reader.read_netcdf()
# reader.visualize(data_type='imp_mat', time_step=6, scale='norm')  # Make sure the path and filename are correctly set
#reader.visualize(data_type='eai_exp', scale='norm')

