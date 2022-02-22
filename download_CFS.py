"""Script adapted for downloading and processing CFS forecast data from AWS. 
Credit of original file to mogismog: https://github.com/mogismog/redissertation/blob/master/redissertation/process_reforecast_data.py
Author: Diego Alfaro
"""

import os
import logging
import pathlib
from typing import Iterable, Dict
from tempfile import TemporaryDirectory
#from sklearn.linear_model import LinearRegression

import s3fs
import numpy as np
import pandas as pd
import cfgrib
import xarray as xr


S3_BUCKET = "noaa-cfs-pds"
BASE_S3_PREFIX = ""
COMMON_COLUMNS_TO_DROP = ["valid_time", "surface"]


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

def create_selection_dict(
    latitude_bounds: Iterable[float],
    longitude_bounds: Iterable[float],
) -> Dict[str, slice]:
    """Generate parameters to slice an xarray Dataset.
    Parameters
    ----------
    latitude_bounds : Iterable[float]
        The minimum and maximum latitude bounds to select.
    longitude_bounds : Iterable[float]
        The minimum and maximum longitudes bounds to select.

    Returns
    -------
    Dict[str, slice]
        A dictionary of slices to use on an xarray Dataset.
    """
    latitude_slice = slice(max(latitude_bounds), min(latitude_bounds))
    longitude_slice = slice(min(longitude_bounds), max(longitude_bounds))
    selection_dict = dict(
        latitude=latitude_slice, longitude=longitude_slice
    )
    return selection_dict


def convert__longitudes(ds: xr.Dataset) -> xr.Dataset:
    """convert longitudes.
    From 0-360 to -180 to 180

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    ds : xr.Dataset
    """
    ds["longitude"] = (
                ("longitude",),
                np.mod(ds["longitude"].values + 180.0, 360.0) - 180.0,
            )
    return ds


def get_clim(
  var_name: str,
  lead_mnth: int,
  tme: xr.DataArray,
  stp: xr.DataArray,
  lat: xr.DataArray,
  lon: xr.DataArray,
  stat: str 
) -> xr.Dataset:
    """Get climatology.
    Parameters
    ----------
    var_name : str
        Local directory to save resulting netCDF file.
    lead_mnth : int
        Lead month of forecast (1 is next month).
    tme : xr.DataArray
        DataArray with dates from forecast
    stp : xr.DataArray
        DataArray with steps from forecast
    lat : xr.DataArray
        DataArray with latitude slice from forecast
    lon : xr.DataArray
        DataArray with longitude slice from forecast
    stat : str
        String specifying climatology statistic: 'clm' or 'std'
    Returns
    -------
    ds : xr.Dataset
        The xarray Dataset with climatological information.
    """
    try:

      assert any((stat=='clm',stat=='std'))
      ds = xr.open_dataset(f'/content/drive/MyDrive/Datos/CFS/{var_name}/{var_name}.f{stat}.l0{lead_mnth}.all.clim.1999.2010.grb2')

      if "soilw" in var_name:
          ds = ds.sel(depthBelowLandLayer=slice(0,.2)).mean("depthBelowLandLayer")
      ds = convert__longitudes(ds)

      ds = ds.isel(time = (ds.time.dt.month == tme.dt.month.values[0]))
      ds = ds.isel(time = (ds.time.dt.day == tme.dt.day.values[0]))
      ds = ds.isel(time = (ds.time.dt.hour == tme.dt.hour.values[0]))

      ds = ds.sel(longitude = slice(np.min(lon.values),np.max(lon.values)),
              latitude = slice(np.max(lat.values),np.min(lat.values)))
      ds=ds.assign(step=stp.values,time=tme)
      ds=ds.drop(['valid_time', 'surface'],errors="ignore")

      return ds
    except Exception as e:
        logging.error(f"Oh no! There was an issue getting climatology {stat}: {e}")
        return


def try_to_open_grib_file(path: str, varnm) -> xr.Dataset:
    """Try to open up a grib file.
    Parameters
    ----------
    path : str
        Path pointing to location of grib file
    Returns
    -------
    ds : xr.Dataset
        The xarray Dataset that contains information
        from the grib file.
    """
    try:
        if "tm" in varnm:
            ds = xr.open_dataset(path, engine="cfgrib", filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2})
        elif "soil" in varnm:
            ds = xr.open_dataset(path, engine="cfgrib", filter_by_keys={"typeOfLevel": "depthBelowLandLayer"}) 
        else:
            ds = xr.open_dataset(path, engine="cfgrib", filter_by_keys={"typeOfLevel": "surface"})
    except Exception as e:
        logger.error(f"Oh no! There was a problem opening up {path}: {e}")
        return
    return ds


def download_and_process_grib(
    s3_prefix: str,
    latitude_bounds: Iterable[float],
    longitude_bounds: Iterable[float],
    save_dir: str,
    var_name: str,
    lead_mnth: int
) -> str:
    """Get a reforecast grib off S3, process, and save locally as netCDF file.
    Parameters
    ----------
    s3_prefix : str
        S3 key/prefix/whatever it's called of a single grib file.
    latitude_bounds : Iterable[float]
        An iterable that contains the latitude bounds, in degrees,
        between -90-90.
    longitude_bounds : Iterable[float]
        An iterable that contains the longitude bounds, in degrees,
        between 0-360.
    save_dir : str
        Local directory to save resulting netCDF file.
    var_name : str
        Local directory to save resulting netCDF file.
    lead_mnth : int
        Lead month of forecast (1 is next month).
    Returns
    -------
    saved_file_path : str
        The location of the saved file.
    """
    base_file_name = s3_prefix.split("/")[-1]
    saved_file_path = os.path.join(save_dir, f"{base_file_name}.nc")
    if pathlib.Path(saved_file_path).exists():
        return saved_file_path

    selection_dict = create_selection_dict(
                latitude_bounds, longitude_bounds)
    logger.info(f"Processing {s3_prefix}")

    fs = s3fs.S3FileSystem(anon=True)
    try:
        with TemporaryDirectory() as t:
            grib_file = os.path.join(t, base_file_name)
            with fs.open(s3_prefix, "rb") as f, open(grib_file, "wb") as f2:
                f2.write(f.read())
            ds = try_to_open_grib_file(grib_file,var_name)
            
            if ds is None:
                return
            ds = ds.sel(selection_dict)[var_name].to_dataset()

            # NOTE: The longitude is originally between 0-360, but
            # for our purpose, we'll convert it to be between -180-180.
            ds = convert__longitudes(ds)
            if "soilw" in var_name:
                ds = ds.sel(depthBelowLandLayer=slice(0,.2)).mean("depthBelowLandLayer")
            ds = ds.expand_dims("time", axis=0).expand_dims("step", axis=1)
            # set data vars to float32
            for v in ds.data_vars.keys():
                ds[v] = ds[v].astype(np.float32)
            ds = ds.drop(COMMON_COLUMNS_TO_DROP, errors="ignore")


            dsclim = get_clim(
              var_name,
              lead_mnth,
              ds.time,
              ds.step,
              ds.latitude,
              ds.longitude,
              "clm"
              )
            
            dsstd = get_clim(
              var_name,
              lead_mnth,
              ds.time,
              ds.step,
              ds.latitude,
              ds.longitude,
              "std"
              )

            ds = ds - dsclim
            ds = ds / dsstd

            directory = f"/content/drive/MyDrive/Datos/CFS/{var_name}/"
            files_in_directory = os.listdir(directory)
            filtered_files = [file for file in files_in_directory if file.endswith(".idx")]
            for file in filtered_files:
              path_to_file = os.path.join(directory, file)
              os.remove(path_to_file)

            ds.to_netcdf(saved_file_path, compute=True)
    except Exception as e:
        logging.error(f"Oh no! There was an issue processing {grib_file}: {e}")
        return
    logging.info(f"All done with {saved_file_path}")
    return saved_file_path


def get_and_process_reforecast_data(
    today,
    forecast_date,
    hour,
    var_name,
    latitude_bounds,
    longitude_bounds,
    local_save_dir,
):
    # let's do some quick checks here...
    if not all([min(latitude_bounds) > -90, max(latitude_bounds) < 90]):
        raise ValueError(
            f"Latitude bounds need to be within -90 and 90, got: {latitude_bounds}"
        )
    if not all([min(longitude_bounds) >= 0, max(longitude_bounds) < 360]):
        raise ValueError(
            f"Longitude bounds must be positive and between 0-360 got: {longitude_bounds}"
        )

    hr_dict = {'prate':'','tmax':'','tmin':''}
    hr = ''#hr_dict[var_name]
    
    save_dir = os.path.join(local_save_dir,var_name)
    save_dir00 = os.path.join(local_save_dir,f'{var_name}00')

    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(save_dir00).mkdir(parents=True, exist_ok=True)

    frcst_mnths = [today + pd.DateOffset(months=i) for i in range(1,7)]

    for i,frcst_mnth in enumerate(frcst_mnths):
        f = f'{S3_BUCKET}/cfs.{forecast_date:%Y%m%d}/{hour}/monthly_grib_01/flxf.01.{forecast_date:%Y%m%d}{hour}.{frcst_mnth:%Y%m}.avrg.grib{hr}.grb2'
        download_and_process_grib(
              f,
              latitude_bounds,
              longitude_bounds,
              save_dir,
              var_name,
              lead_mnth = i+1
        )
####TODO: cambiar lead time para cambio de mes
    if hour == "00":
        for nfrct in range(2, 5):
            for i,frcst_mnth in enumerate(frcst_mnths[0:3]):
                f = f'{S3_BUCKET}/cfs.{forecast_date:%Y%m%d}/{hour}/monthly_grib_0{nfrct}/flxf.0{nfrct}.{forecast_date:%Y%m%d}{hour}.{frcst_mnth:%Y%m}.avrg.grib{hr}.grb2'
                download_and_process_grib(
                  f,
                  latitude_bounds,
                  longitude_bounds,
                  save_dir00,
                  var_name,
                  lead_mnth = i+1
            )
            

    # month=f"{mnth:02}"
    # ds = xr.open_mfdataset(os.path.join(local_save_dir,month, "*.nc"), combine="by_coords")
    # final_path = os.path.join(final_save_path,var_names[0]+"_"+str(min(forecast_days_bounds))+
    #                                   "day_"+month+".nc")
    # ds.load().to_netcdf(final_path, mode="w", compute=True)
    # !cp $final_path $drive_path  #Copia a Drive
    
    logger.info("All done!")


if __name__ == "__main__":
  today = pd.Timestamp('today')
  a = 5
  for i in range(0,a+1):
      date = today + pd.Timedelta(f'{i-a}D')
      for j in range(0,4):
          hour = str(f'{j*6:02}')
          get_and_process_reforecast_data(today, date, hour, 
                                          "prate",#"soilw",
                                          (33, 14),(241, 275),
                                          "./CFS_frcst")

