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

from osgeo import gdal, osr
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LogNorm

class ImpactWriteGeoTIFF:
    """
    A class to write Impact data to a GeoTIFF file.

    Attributes:
        impact_object (Impact): The impact object containing the data to be written.
    """

    def __init__(self, impact_object):
        """
        Initialize the ImpactWriteGeoTIFF with an impact object.
        
        Args:
            impact_object (Impact): An object containing impact data like coordinates and values.
        """
        self.impact_object = impact_object

    def write_to_geotiff(self, filename):
        """
        Write the impact data to a GeoTIFF file.
        
        Args:
            filename (str): The name of the file where the GeoTIFF data will be written.
        """
        # Extract coordinates and impact values from the impact object
        coords = self.impact_object.coord_exp
        eai_exp_values = self.impact_object.eai_exp

        # Create a dictionary to hold summed eai_exp values and counts for each unique (lat, lon) pair
        eai_exp_dict = {}
        count_dict = {}
        
        # Sum and count eai_exp values for each unique coordinate
        for i in range(len(coords)):
            lat, lon = coords[i]
            key = (lat, lon)
            if key in eai_exp_dict:
                eai_exp_dict[key] += eai_exp_values[i]
                count_dict[key] += 1
            else:
                eai_exp_dict[key] = eai_exp_values[i]
                count_dict[key] = 1

        # Average the eai_exp values for each coordinate
        for key in eai_exp_dict:
            eai_exp_dict[key] /= count_dict[key]

        # Generate arrays for unique latitudes and longitudes
        unique_lats = np.unique(coords[:, 0])
        unique_lons = np.unique(coords[:, 1])

        # Initialize an array for reshaped eai_exp values
        eai_exp_reshaped = np.zeros((len(unique_lats), len(unique_lons)))

        # Reshape and assign eai_exp values to the array
        for i, lat in enumerate(unique_lats):
            for j, lon in enumerate(unique_lons):
                key = (lat, lon)
                if key in eai_exp_dict:
                    eai_exp_reshaped[i, j] = eai_exp_dict[key]

        # Flip the array along the latitude axis for correct geographical representation
        eai_exp_reshaped = np.flipud(eai_exp_reshaped)

        # Create a new GeoTIFF file
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
            filename,
            len(unique_lons),
            len(unique_lats),
            1,
            gdal.GDT_Float32
        )

        # Set the geotransform for the dataset
        dataset.SetGeoTransform([
            unique_lons.min(),  # Origin longitude
            (unique_lons.max() - unique_lons.min()) / len(unique_lons),  # Pixel size in longitude
            0,
            unique_lats.max(),  # Origin latitude
            0,
            -(unique_lats.max() - unique_lats.min()) / len(unique_lats)  # Pixel size in latitude (negative)
        ])

        # Set the projection to WGS84
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dataset.SetProjection(srs.ExportToWkt())

        # Write the eai_exp data to the raster band
        dataset.GetRasterBand(1).WriteArray(eai_exp_reshaped)
        dataset.FlushCache()  # Ensure data is written to disk

class ImpactReaderGeoTIFF:
    """
    A class to read and visualize Impact data from a GeoTIFF file.

    Attributes:
        filename (str): Path to the GeoTIFF file.
    """

    def __init__(self, filename):
        """
        Initialize the ImpactReaderGeoTIFF with a GeoTIFF file path.
        
        Args:
            filename (str): The path to the GeoTIFF file containing impact data.
        """
        self.filename = filename

    def plot_geotiff(self, scale='normal'):
        """
        Plot the GeoTIFF file data.
        
        Args:
            scale (str): The scale type for data visualization, either 'normal' or 'log'.
        """
        # Open the GeoTIFF file and read data
        with rasterio.open(self.filename) as src:
            data, bounds = self._read_data(src)
            fig, ax = self._create_plot()
            self._plot_data(ax, data, bounds, scale)
            self._add_features(ax)
            self._add_labels_and_ticks(ax, bounds)
            plt.show()

    def _read_data(self, src):
        """
        Read data from the source GeoTIFF file.
        
        Args:
            src: The rasterio file object.
            
        Returns:
            Tuple containing the data array and the bounds of the data.
        """
        data = src.read(1)
        bounds = src.bounds
        return data, bounds

    def _create_plot(self):
        """
        Create a matplotlib plot with a specific size and cartographic projection.
        
        Returns:
            The figure and axes objects of the plot.
        """
        return plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    def _plot_data(self, ax, data, bounds, scale):
        """
        Plot the data on the provided axes.
        
        Args:
            ax: The axes object for the plot.
            data: The data array to be plotted.
            bounds: The geographical bounds of the data.
            scale: The scale type for data visualization.
        """
        # Set the normalization and label based on the scale type
        if scale == 'log':
            norm = LogNorm()
            label = 'eai_exp (Log Scale)'
        else:
            norm = None
            label = 'eai_exp'
        
        # Create a meshgrid for X and Y coordinates
        ny, nx = data.shape
        x = np.linspace(bounds.left, bounds.right, nx)
        y = np.linspace(bounds.top, bounds.bottom, ny)
        X, Y = np.meshgrid(x, y)
        
        # Plot the data using pcolormesh
        sc = ax.pcolormesh(X, Y, data, cmap='RdYlBu_r', norm=norm)
        cbar = plt.colorbar(sc, ax=ax, pad=0.05)  # Add a colorbar
        cbar.set_label(label)  # Label the colorbar

    def _add_features(self, ax):
        """
        Add geographical features to the plot for better context.
        
        Args:
            ax: The axes object for the plot.
        """
        ax.coastlines()  # Add coastlines
        ax.gridlines()   # Add gridlines
        ax.add_feature(cfeature.BORDERS, linestyle=':')  # Add country borders

    def _add_labels_and_ticks(self, ax, bounds):
        """
        Add labels, titles, and ticks to the plot for better readability.
        
        Args:
            ax: The axes object for the plot.
            bounds: The geographical bounds of the data.
        """
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Expected Annual Impact')
        
        # Set ticks for longitude and latitude
        num_ticks = 10
        x_ticks = np.linspace(bounds.left, bounds.right, num_ticks)
        y_ticks = np.linspace(bounds.bottom, bounds.top, num_ticks)
        
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels([f"{x:.2f}" for x in x_ticks])
        ax.set_yticklabels([f"{y:.2f}" for y in y_ticks])
