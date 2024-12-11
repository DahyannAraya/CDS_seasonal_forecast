#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File to download seasonal forecast data from cds
Aquivalent to what is done in jupyter notebook download_read_grib_calc_indices.ipynb,
except that it does not calculate indices
but if lots of data has to be downloaded this can be run in background using screen:
> screen
> module load conda
> source activate cds_climada_grib
> python download_grib_data_cds.py

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
# define variables etc. to be downloaded
variables=['2m_dewpoint_temperature', '2m_temperature']
# start of filename which indicates what kind of variables are in file
filename_lead='2m_temps'

startyr=2017
endyr=2017
month_list=['01', '02', '03', '04', '05', '06', '07', '08', '09']
area=[90, -180, -90, 180]
area_str=f'{int(area[0])}_{int(area[1])}_{int(area[2])}_{int(area[3])}'

# Flag to force overwriting of daily and index files which already exist if set to True
overwrite=False

# set the path were you want to store the downloaded data
from climada import CONFIG
import cdsapi
DATA_OUT=str(CONFIG.data_dir)

# -------------------------------------------------
# Getting libraries and utilities
# -------------------------------------------------
import os
import cdsapi
import logging

import cfgrib
import xarray as xr

from util_functions import calc_min_max_lead, download_multvar_multlead 

# -------------------------------------------------
# Create a simple logger
# -------------------------------------------------

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO)
logger = logging.getLogger()

# -------------------------------------------------
# create directory if it does not exist yet
# -------------------------------------------------

if not os.path.exists(DATA_OUT):
    os.makedirs(DATA_OUT)

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
        # for hindcast download data for ~4 months worth of leadtimes (so we can disregard the first month but still have 3 month ahead)
        max_lead_month = 4
    else:
        # for forecast download data for ~6 months worth of leadtimes
        max_lead_month = 6
    
    for month in month_list:
        logger.info(f'Month: {month}')
        out_dir=f'{DATA_OUT}/grib/{year}/{month}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        min_lead, max_lead = calc_min_max_lead(year, int(month), max_lead_month)

        leadtimes=[*range(min_lead, max_lead, 6)]
        logger.info(f'{len(leadtimes)} leadtimes to download.')
        logger.debug(f'which are: {leadtimes}')

        download_file=f'{out_dir}/{filename_lead}_{area_str}_{year}{month}.grib'
        download_multvar_multlead(download_file, variables, year, month, leadtimes, area, overwrite)

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
            # calculate mean over 4 time steps
            ds_daily = ds.coarsen(step=4, boundary='trim').mean()
            # save data to netcdf   
            ds_daily.to_netcdf(daily_file_out)