import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
from matplotlib.colors import LogNorm
import cartopy.feature

class ImpactReaderNetCDF:
    def __init__(self, filename):
        self.filename = filename

    def read_netcdf(self):
        self.ds = xr.open_dataset(self.filename)
        self.lat = self.ds['latitude'].values
        self.lon = self.ds['longitude'].values

    def visualize_eai_exp(self, scale='normal'):
        data = self.ds['eai_exp'].values
        self._visualize(data, 'eai_exp', scale)

    def visualize_imp_mat(self, year, scale='normal'):
        time_index = self.ds['time'].values == np.datetime64(f'{year}')
        data = self.ds['imp_mat'].sel(time=time_index).mean(dim='time').values
        self._visualize(data, 'imp_mat', scale)

    def _visualize(self, data, title, scale):
        fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        
        if scale == 'log':
            norm = LogNorm()
            label = f'{title} (Log Scale)'
        else:
            norm = None
            label = title
        
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

def visualize_impact_data(filename, year=None, scale='log'):
    reader = ImpactReaderNetCDF(filename)
    reader.read_netcdf()
    reader.visualize_eai_exp(scale=scale)
    if year is not None:
        reader.visualize_imp_mat(year=year, scale=scale)

# Example usage:
#visualize_impact_data('impact_data_F.nc', year=2010, scale='log')
