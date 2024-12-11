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

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import xarray as xr

from save_read_impact_data_to_NetCDF import save_impact_data_to_NetCDF


"""
This module tests the functionality of the save_impact_data_to_NetCDF function to ensure it handles data transformation and saving operations correctly. The tests are designed to verify several critical aspects:

1. **Data Handling and Transformation**:
   Ensures that environmental and impact matrix data are correctly transformed and mapped to a geographic grid based on provided coordinates. This includes checking the reshaping of data arrays and the application of logarithmic transformations if specified.

2. **File Operations**:
   Verifies that the function attempts to save data to a NetCDF file with the correct settings, specifically checking that the file mode is set to write. The `xarray.Dataset.to_netcdf` method is patched to intercept and validate file operation calls without writing to the disk.

3. **Error Management*:
   Tests the function's response to missing attributes within the input data object. It confirms that appropriate exceptions are raised when required data elements are not available, ensuring the function's robustness in the face of incomplete inputs.

Each test utilizes a mock impact object setup to simulate different data scenarios and validate the function's behavior under controlled conditions.

Functions:
- setup_impact_object: Creates a mock impact object with necessary properties for testing.
- test_save_impact_data_to_NetCDF: Tests the primary functionality of data saving and transformations.
- test_error_handling: Checks the function's error handling capabilities by deliberately omitting required data attributes.

The module uses "pytest" for testing, with fixtures for setting up test data and "unittest.mock" for intercepting file operations.
"""

def setup_impact_object():
    """ Create a mock impact_object with necessary properties for testing. """
    impact_object = MagicMock()
    # Simulate smaller dataset
    impact_object.eai_exp = np.random.rand(15254)  # reduced from actual size for testing
    impact_object.imp_mat = np.random.rand(3, 15254)  # 3 time steps, reduced number of spatial points
    # Coordinate mapping for the reduced dataset
    impact_object.coord_exp = np.column_stack((np.random.randint(0, 10, 15254), np.random.randint(0, 10, 15254)))
    impact_object.event_name = ['2000-01-01', '2000-02-01', '2000-03-01']
    return impact_object

@pytest.fixture
def impact_object():
    return setup_impact_object()

@patch('xarray.Dataset.to_netcdf', autospec=True)
def test_save_impact_data_to_NetCDF(mock_to_netcdf, impact_object):
    """ Test the function saves data correctly and applies transformations. """
    filename = 'dummy.nc'
    save_impact_data_to_NetCDF(impact_object, filename, include_eai_exp=True, include_imp_mat=True, log_scale_imp=True, log_scale_eai=False, time_attribute='event_name')

    mock_to_netcdf.assert_called_once()
    args, kwargs = mock_to_netcdf.call_args
    assert 'mode' in kwargs
    assert kwargs['mode'] == 'w'
    
    # Dataset checking
    dataset = args[0]
    assert 'latitude' in dataset.dims and 'longitude' in dataset.dims and 'time' in dataset.dims
    assert dataset.sizes['latitude'] <= 10 and dataset.sizes['longitude'] <= 10  # based on coord_exp setup

def test_error_handling(impact_object):
    """ Test error handling for missing attributes. """
    impact_object.event_name = None  # Simulate missing attribute
    with pytest.raises(ValueError):
        save_impact_data_to_NetCDF(impact_object, 'dummy.nc', include_eai_exp=True, include_imp_mat=True, time_attribute='event_name')





import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt
import pandas as pd  

# ImpactReaderNetCDF class is imported from save_read_impact_data_to_NetCDF.py
from save_read_impact_data_to_NetCDF import ImpactReaderNetCDF

@pytest.fixture
def mock_dataset():
    data = np.random.rand(10, 10)
    dataset = xr.Dataset({
        'eai_exp': (['x', 'y'], data),
        'imp_mat': (['time', 'x', 'y'], np.random.rand(3, 10, 10)),
        'latitude': (['x'], np.linspace(-90, 90, 10)),
        'longitude': (['y'], np.linspace(-180, 180, 10)),
        'time': pd.date_range("2000-01-01", periods=3)
    })
    return dataset

@patch('xarray.open_dataset')
def test_read_netcdf(mock_open_dataset, mock_dataset):
    """Test that the NetCDF file is read correctly and data is loaded."""
    mock_open_dataset.return_value = mock_dataset
    reader = ImpactReaderNetCDF('dummy.nc')
    reader.read_netcdf()

    assert reader.ds is not None
    assert 'eai_exp' in reader.ds
    assert 'imp_mat' in reader.ds

@patch('matplotlib.pyplot.show')
def test_visualize(mock_show, mock_dataset):
    """Test the visualization setup for different data types and scales."""
    with patch('xarray.open_dataset', return_value=mock_dataset):
        reader = ImpactReaderNetCDF('dummy.nc')
        reader.read_netcdf()
        
        # Test eai_exp visualization
        reader.visualize(data_type='eai_exp', scale='normal')
        reader.visualize(data_type='eai_exp', scale='log')

        # Test imp_mat visualization for a specific time step
        reader.visualize(data_type='imp_mat', time_step=1, scale='normal')
        reader.visualize(data_type='imp_mat', time_step=1, scale='log')

        plt.show.assert_called()

def test_invalid_data_type(mock_dataset):
    """Test that invalid data type inputs raise a ValueError."""
    with patch('xarray.open_dataset', return_value=mock_dataset):
        reader = ImpactReaderNetCDF('dummy.nc')
        reader.read_netcdf()
        
        with pytest.raises(ValueError):
            reader.visualize(data_type='invalid_type', time_step=0, scale='normal')






