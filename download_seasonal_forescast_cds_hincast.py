#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This python script downloads seasonal forecast data from cds, calculates daily means
and calculates heat indices which can be used in CLIMADA.
Aquivalent to what is done in jupyter notebook download_read_grib_calc_indices.ipynb
but if lots of data has to be downloaded this can be run in background using screen:
Usage @IAC servers:
> screen
> module load conda
> source activate cds_climada_grib
> python download_seasonal_forecast_cds_calc_index_thermofeel_all_years_months.py

This script specifically accesses daily and subdaily Seasonal forecast data at single 
levels,from Deutscher Wetterdienst (DWD), GCFS2.0. The dataset includes real-time 
forecasts from 2017 onwards,featuring 50 ensemble members, and retrospective forecasts 
(hindcasts) for the period from 1993-2016, including 30 members. Real-time forecasts 
are issued monthly on the 6th at 12UTC for ECMWF and on the 10th at 12 UTC for other 
centers, covering a forecast period of 7 months. The data, providing a 1° x 1° horizontal
 resolution, spans globally and is available in GRIB format. For more information, visit: 
https://cds.climate.copernicus.eu/cdsapp#!/dataset/seasonal-original-single-levels?tab=overview

For more details about DWD seasonal forecast data, consult "The German Climate Forecast System: GCFS":
https://doi.org/10.1029/2020MS002101
"""
# *******************************************************************************
#                         U S E R  *  O P T I O N S
# *******************************************************************************

# define index to be calculated using thermofeel package
tf_index = 'Tmean' # HIS, HIA or Tmean
index_long_name = 'heat_index_adjusted'

# define variables etc. to be downloaded
variables = ['2m_dewpoint_temperature', '2m_temperature']
# start of filename which indicates what kind of variables arein file
filename_lead='2m_temps'

startyr=2017 #1993
endyr=2023 #2016
month_list=['01', '02', '03', '04', '05', '06', '07', '08', '09']
area=[90, -180, -90, 180]
area_str=f'{int(area[0])}_{int(area[1])}_{int(area[2])}_{int(area[3])}'
hincast_ini = 1992
hicast_end = 2017

# Flag to force overwriting of daily and index files which already exist if set to True
overwrite=False

# set the path were you want to store the downloaded data
#from climada import CONFIG
#import cdsapi
#DATA_OUT = str(CONFIG.data_dir)
import os
DATA_OUT = "/wcr/daraya/data/seasonal_forecast_dwd/" # create a file to save the data
os.makedirs(DATA_OUT, exist_ok=True)

# define path to write calculated ouput data
index_out = f"{DATA_OUT}/{tf_index}"

# -------------------------------------------------
# Getting libraries and utilities
# -------------------------------------------------
import os
import sys
import cdsapi
import logging
from nco import Nco
nco = Nco()

import xarray as xr
import thermofeel as tf
import matplotlib.pyplot as plt

from util_functions import calc_min_max_lead_da, download_multvar_multlead, calculate_thermofeel_index

# -------------------------------------------------
# Create a simple logger
# -------------------------------------------------

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger()

# -------------------------------------------------
# create directory if do not exist yet
# -------------------------------------------------

if not os.path.exists(DATA_OUT):
    os.makedirs(DATA_OUT)

if (os.access(index_out, os.F_OK) == False):
    os.makedirs(index_out)

# read list of corrupt files
with open('corrupt_files.txt', 'r') as file:
    # Read the file line by line and store each line as an item in a list
    corrupt_files = [line.strip() for line in file.readlines()]
logger.info(corrupt_files)


# -------------------------------------------------
# download data for each year between startyr and endyr
# for each hindcast/forecast starting in months in month_list
# for all variables and leadtimes.
# Calculate daily means and save as netcdf
# -------------------------------------------------
for year in range(startyr, endyr+1):
    logger.info(f'Year: {year}')
    # for every month we can download data in 6-hourly forecast steps for up to ~6 months
    if year <= 2016:
        # for hindcast download data for ~4 months worth of leadtimes
        # (so we can disregard the first month but still have 3 month ahead)
        max_lead_month = 1
    else:
        # for forecast download data for ~6 months worth of leadtimes
        max_lead_month = 3 #6
    
    for month in month_list:
        logger.info(f'Month: {month}')
        out_dir=f'{DATA_OUT}/grib/{year}/{month}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        min_lead, max_lead = calc_min_max_lead_da(year, int(month), max_lead_month)

        leadtimes=[*range(min_lead, max_lead, 6)]
        logger.info(f'{len(leadtimes)} leadtimes to download.')
        logger.debug(f'which are: {leadtimes}')

        download_file=f'{out_dir}/{filename_lead}_{area_str}_{year}{month}.grib'
        if download_file in corrupt_files:
            continue

        download_multvar_multlead(download_file, variables, year, month, leadtimes, area=area, overwrite=overwrite)

        # calculate daily mean
        daily_out=f'{DATA_OUT}/netcdf/daily/{year}/{month}'
        if not os.path.exists(daily_out):
            os.makedirs(daily_out)
        daily_file_out = f'{daily_out}/{filename_lead}_{area_str}_{year}{month}.nc'

        if not os.path.isfile(daily_file_out) or overwrite:
            # read grib file
            try:
                ds = xr.load_dataset(download_file, engine="cfgrib")
            except FileNotFoundError:
                logger.error('Grib file does not exist, download failed.')
                continue
            # calculate mean over 4 timesteps
            ds_daily = ds.coarsen(step=4, boundary='trim').mean()
            # write data to netcdf      
            ds_daily.to_netcdf(daily_file_out)


# ------------------------------------------------- 
# calculate heat indices using thermofeel package
# for all years, all months in month_list
# -------------------------------------------------

The funtion primary Outputs are - Index (tmean, HIS) NetCDF Files:
T mean', the daily values in the index files represent for example the mean temperature for each day, for individual ensemble members 
Ensemble Statistics Files: While based on daily values, the statistics themselves (mean, max, min, std) are aggregated measures across all ensemble members for each day. The output, therefore, provides a daily statistical summary rather than individual daily values from each ensemble membe

for year in range(startyr, endyr+1):
    if not os.path.exists(f'{index_out}/{year}'):
        os.makedirs(f'{index_out}/{year}')
    # for all months
    for month in month_list:
        index_file = f'{index_out}/{year}/{tf_index}_{area_str}_{year}{month}.nc'
        # do not calculate index if it has already been calculated and saved
        if os.path.isfile(f'{index_file}') and not overwrite:
            logger.info(f'Corresponding index file {index_file} already exists!')
            ds_index = xr.load_dataset(index_file)
        else:
            daily_out = f'{DATA_OUT}/netcdf/daily/{year}/{month}'
            try:
                with xr.open_dataset(f'{daily_out}/{filename_lead}_{area_str}_{year}{month}.nc') as daily_ds:
                    if tf_index != 'Tmean':
                        # read daily data
                        da_t2k = daily_ds['t2m']
                        da_tdk = daily_ds['d2m']
                        da_index = calculate_thermofeel_index(da_t2k, da_tdk, tf_index)
                    else:
                        # Tmean index was basically calculated when calculating daily means, change unit from Kelvin to degrees Celsius
                        da_index = daily_ds['t2m'] - 275.15
                        da_index.attrs["units"] = "degC"
            except FileNotFoundError:
                logger.warning(f'Data file {daily_out}/{filename_lead}_{area_str}_{year}{month}.nc does not exist!')
                continue
            # put data array into dataset which can be saved as netcdf
            ds_index = xr.Dataset({})
            ds_index[f'{tf_index}'] = da_index
            # write index to netcdf
            ds_index.to_netcdf(f'{index_file}')
        
        # calculate some statistics over all ensemble members
        da_index = ds_index[tf_index]
        da_index_ens_mean = da_index.mean('number')
        da_index_ens_max = da_index.max('number')
        da_index_ens_min = da_index.min('number')
        da_index_ens_std = da_index.std('number')

        # put ensemble statistics into dataset
        ds_statistics = xr.Dataset(
            dict(ensemble_mean = da_index_ens_mean,
             ensemble_max = da_index_ens_max,
             ensemble_min = da_index_ens_min,
             ensemble_std = da_index_ens_std
                ),
            attrs=dict(
                description=f"{index_long_name} ensemble statistics",
                units=da_index.units,
            ),
        )

        # write dataset to netcdf
        ds_statistics.to_netcdf(f"{index_out}/{year}/{tf_index}_{area_str}_{year}{month}_statistics.nc")
        logger.info(f'Data was saved to {index_out}/{year}/{tf_index}_{area_str}_{year}{month}_statistics.nc')
        

# ------------------------------------------------- 
# Input CLIMADA intensity.calculate total heat 
# waves days per month
# for all years, and save the statisc of all the 
# member as nc. Input CLIMADA intensity 
# -------------------------------------------------
#  This module calculates the total number of heatwave days per month across all years
#  within a specified range and saves the statistics for all ensemble members as a NetCDF file.
#  A heatwave is defined as a period of at least three consecutive days with temperatures exceeding
#  27 degrees Celsius. The script processes temperature data, identifies heatwave events, calculates
#  statistical measures across ensemble members, and organizes the output in a structured format.

#  The funtion primary Output - Heat Wave Statistics NetCDF Files:
#  File Name: stats_file - This is constructed dynamically for each year and month combination within the specified range (startyr to endyr). 
# Content:
#  ensemble_mean: The average number of heatwave days across all ensemble members for each month.
#  ensemble_max: The maximum number of heatwave days observed among all ensemble members for each month.
#  ensemble_min: The minimum number of heatwave days observed among all ensemble members for each month.
#  ensemble_std: The standard deviation of the number of heatwave days across all ensemble members for each month.


# Define the threshold temperature for a heat wave
threshold_temp = 27.0

def calculate_heat_wave_days(da_temperature):
    """
    """
    # Identify days with temperatures above the threshold
    above_threshold = da_temperature > threshold_temp

    # Use a rolling window over the 'step' dimension to identify consecutive days above the threshold
    rolling_sum = above_threshold.rolling(step=3).sum() >= 3

    # Calculate total heat wave days in a month
    heat_wave_days = rolling_sum.sum(dim='step')

    return heat_wave_days

# Define the base output directory
base_output_folder = DATA_OUT
heat_wave_folder = os.path.join(base_output_folder, 'heat_wave_days')

# Initialize logger
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger()

# Loop through each year and month
for year in range(startyr, endyr + 1):
    # Create the heat wave folder for the specific year if it doesn't exist
    year_heat_wave_folder = os.path.join(heat_wave_folder, str(year))
    os.makedirs(year_heat_wave_folder, exist_ok=True)

    for month in month_list:
        index_file = f'{index_out}/{year}/{tf_index}_{area_str}_{year}{month}.nc'
        stats_file = os.path.join(year_heat_wave_folder, f"heat_wave_stats_{year}{month}.nc")

        # Check if the statistics file already exists
        if os.path.exists(stats_file):
            logger.info(f"Statistics file {stats_file} already exists. Skipping calculation.")
            continue
        try:
            # Load the pre-calculated Tmean dataset
            ds_index = xr.load_dataset(index_file)
            da_tmean = ds_index[tf_index]  # Convert from Kelvin to Celsius if necessary

            # Calculate the heat wave days
            heat_wave_days = calculate_heat_wave_days(da_tmean)

            # Calculate statistics over ensemble members
            ensemble_mean = heat_wave_days.mean(dim='number')
            ensemble_max = heat_wave_days.max(dim='number')
            ensemble_min = heat_wave_days.min(dim='number')
            ensemble_std = heat_wave_days.std(dim='number')

            # Create a dataset for the statistics
            ds_heat_wave_stats = xr.Dataset({
                'ensemble_mean': ensemble_mean,
                'ensemble_max': ensemble_max,
                'ensemble_min': ensemble_min,
                'ensemble_std': ensemble_std
            })

             # Create the heat wave folder for the specific year if it doesn't exist
            year_heat_wave_folder = os.path.join(heat_wave_folder, str(year))
            os.makedirs(year_heat_wave_folder, exist_ok=True)

            # Save the statistics to a NetCDF file
            ds_heat_wave_stats.to_netcdf(stats_file)
            logger.info(f'Heat wave statistics saved to {stats_file}')

        except FileNotFoundError as e:
            logger.warning(f'File not found: {e.filename}')
        except Exception as e:
            logger.error(f'An error occurred: {e}')


# ------------------------------------------------- 
# calculate climatology heat waves days per month
# for the hincast, and save the statisc of all 
# the member as nc. Input CLIMADA threshold
# -------------------------------------------------
# # Define a dictionary to hold heat wave days data for each month
monthly_heat_wave_data = {month: [] for month in month_list}

# Loop through each year and month
for year in range(startyr, endyr + 1):
    if year > 2016:
        break  # Skip years after 2016 for hindcast data
    for month in month_list:
        index_file = f'{index_out}/{year}/{tf_index}_{area_str}_{year}{month}.nc'
        try:
            # Load the Tmean dataset
            ds_index = xr.load_dataset(index_file)
            da_tmean = ds_index[tf_index]

            # Calculate heat wave days
            heat_wave_days = calculate_heat_wave_days(da_tmean)

            # Aggregate data
            monthly_heat_wave_data[month].append(heat_wave_days.mean(dim='number'))  # Storing the mean over ensemble members

        except FileNotFoundError:
            logger.warning(f'File not found: {index_file}')
        except Exception as e:
            logger.error(f'An error occurred: {e}')
#
# Directory for saving monthly statistics
monthly_stats_folder = os.path.join(base_output_folder, 'monthly_heat_wave_stats_hincast')
os.makedirs(monthly_stats_folder, exist_ok=True)

# Calculate statistics for each month and save
for month in month_list:
    stats_file = os.path.join(monthly_stats_folder, f"monthly_heat_wave_stats_hincast{month}.nc")

    # Check if the file already exists
    if os.path.exists(stats_file):
        logger.info(f"Statistics file {stats_file} already exists. Skipping calculation.")
        continue

    # Combine data for the month across all years
    combined_data = xr.concat(monthly_heat_wave_data[month], dim='year')

    # Calculate statistics
    monthly_mean = combined_data.mean(dim='year')
    monthly_max = combined_data.max(dim='year')
    monthly_min = combined_data.min(dim='year')
    monthly_std = combined_data.std(dim='year')

    # Create a dataset for monthly statistics
    ds_monthly_stats = xr.Dataset({
        'monthly_mean': monthly_mean,
        'monthly_max': monthly_max,
        'monthly_min': monthly_min,
        'monthly_std': monthly_std
    })

    # Save to NetCDF
    ds_monthly_stats.to_netcdf(stats_file)
    logger.info(f'Monthly heat wave statistics hicast for {month} saved to {stats_file}')


import xarray as xr
import pandas as pd
import os
from pathlib import Path

# ------------------------------------------------- 
# Combine Heatwave Statistics into a Single NetCDF File
# -------------------------------------------------
"""
This script combines individual NetCDF files containing heatwave statistics for specific months and years into a single comprehensive NetCDF file, facilitating analysis over a defined temporal range.

Steps:
1. Setup: Defines the directory for individual files and the output path for the combined dataset.
2. Time Range: Specifies the years and months for data aggregation.
3. Data Aggregation:
   - Initializes a list for datasets.
   - Iterates over each specified year and month, opening existing NetCDF files, adding a 'time' dimension, and appending them to the list.
   - Logs skipped files if not found.
4. Concatenation and Saving:
   - If datasets are present, concatenates them along the 'time' dimension.
   - Saves the combined dataset to the output file path.
   - Logs a message if no files were added for concatenation.

Outcome:
Produces a unified NetCDF file aggregating heatwave statistics, enabling comprehensive trend analysis.

Usage:
Intended for researchers and analysts for studying heatwave impacts and informing adaptation strategies.
"""
# -------------------------------------------------
#  This module aggregates heatwave days statistics per month across all years within a specified range, saving the ensemble members' statistics as a NetCDF file.
#  Heatwaves are defined as periods of at least three consecutive days exceeding a temperature threshold. It processes temperature data, identifies heatwave events, and calculates statistical measures across ensemble members.

# Primary Output - Heat Wave Statistics NetCDF Files:
#  File Name Pattern: heat_wave_stats_{year}{month}.nc for each year and month within the specified range.
# Content:
#  ensemble_mean: Monthly average of heatwave days across ensemble members.
#  ensemble_max: Monthly maximum of heatwave days observed among ensemble members.
#  ensemble_min: Monthly minimum of heatwave days observed among ensemble members.
#  ensemble_std: Monthly standard deviation of heatwave days across ensemble members.


# Define the base directory and the output file path
base_dir = "/wcr/daraya/data/seasonal_forecast_dwd/heat_wave_days/"
output_file = "/wcr/daraya/data/seasonal_forecast_dwd/combined_heat_wave_forecast.nc"

# Define the range of years and the list of months to process
years = range(2017, 2024)
month_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09']

# Prepare a list to hold datasets
datasets = []

for year in years:
    for month in month_list:
        file_path = f"{base_dir}/{year}/heat_wave_stats_{year}{month}.nc"
        if os.path.exists(file_path):
            ds = xr.open_dataset(file_path)
            # Add a 'time' dimension based on the year and month
            ds['time'] = pd.to_datetime(f'{year}-{month}-01')
            datasets.append(ds)
        else:
            print(f"File not found, skipped: {file_path}")

if datasets:
    # Concatenate all datasets along the 'time' dimension
    combined_ds = xr.concat(datasets, dim='time')
    combined_ds.to_netcdf(output_file)
    print(f"Combined dataset saved to {output_file}")
else:
    print("No files were found or added for concatenation.")



# Example usage
if __name__ == "__main__":
    base_dir = "/Users/daraya/Documents/seasonal_forecast_cds_git/seasonal_forecast_dwd/heat_wave_days"
    output_file = "/Users/daraya/Documents/seasonal_forecast_cds_git/seasonal_forecast_dwd/combined_heat_wave_forecast.nc"
    start_year = 2015
    end_year = 2016
    month_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09']

    combine_heatwave_statistics(base_dir, output_file, start_year, end_year, month_list)