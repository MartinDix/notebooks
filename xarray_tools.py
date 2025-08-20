
# Functions for xarray

import numpy as np, xarray as xr, warnings

# Should be simplified in the next release https://github.com/pydata/xarray/pull/3935/files
# http://xarray.pydata.org/en/stable/examples/monthly-means.html
dpm = {'noleap': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '365_day': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'standard': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'proleptic_gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'all_leap': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '366_day': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '360_day': [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]}


def leap_year(year, calendar='standard'):
    """Determine if year is a leap year"""
    leap = False
    if ((calendar in ['standard', 'gregorian',
        'proleptic_gregorian', 'julian']) and
        (year % 4 == 0)):
        leap = True
        if ((calendar == 'proleptic_gregorian') and
            (year % 100 == 0) and
            (year % 400 != 0)):
            leap = False
        elif ((calendar in ['standard', 'gregorian']) and
                 (year % 100 == 0) and (year % 400 != 0) and
                 (year > 1583)):
            leap = False
    return leap

# def get_dpm(time, calendar='standard'):
#     """
#     Return a array of days per month corresponding to the months provided in `months`
#     Perhaps check against the time bounds for validation.
#     """
#     month_length = np.zeros(len(time), dtype=np.int)

#     cal_days = dpm[calendar]

#     for i, (month, year) in enumerate(zip(time.month, time.year)):
#         month_length[i] = cal_days[month]
#         if leap_year(year, calendar=calendar) and month == 2:
#             month_length[i] += 1
#     return month_length

def annual_mean(ds, original_time_axis=False):
    """ Properly month length weighted annual mean of a DataArray"""
    if not ds.time.dt.month[0] == 1:
        raise ValueError("Data does not start with January")
    if not ds.time.dt.month[1] == 2:
        raise ValueError("Non-monthly data")
    # Process only complete years
    nmon = 12*(len(ds.time)//12)
    month_length = ds.time.dt.days_in_month[:nmon]
    # Eventually use weighted ??? https://github.com/pydata/xarray/issues/3937
    weights = month_length.groupby('time.year') / month_length.groupby('time.year').sum()
    ann_mean = (ds[:nmon]*weights).groupby('time.year').sum(dim='time',min_count=1)
    if original_time_axis:
        ann_mean = ann_mean.assign_coords({'year':ds[:nmon].time.groupby('time.year').mean(dim='time')})
    return ann_mean

def seasonal_mean(ds):
    month_length = ds.time.dt.days_in_month
    result = ((ds * month_length).resample(time='QS-DEC').sum() /
          month_length.resample(time='QS-DEC').sum())
    return result

def global_mean(ds,weights):
    """ Area weighted global mean of a DataArray"""
    # Check the shapes match
    assert ds.shape[-2:] == weights.shape, 'Shape mismatch in global_mean'
    assert ds.dims[-2:] == weights.dims, 'Coordinate mismatch in global_mean'
    dims = weights.dims
    if (weights[dims[0]]==ds[dims[0]]).all() and (weights[dims[1]]==ds[dims[1]]).all():
        ds_w = ds.weighted(weights)
    elif np.allclose(weights[dims[0]].data,ds[dims[0]].data) and np.allclose(weights[dims[1]].data,ds[dims[1]].data):
        ds_w = ds.weighted(weights.reindex_like(ds, method='nearest', tolerance=1e-4))
        # Warnings don't work properly?
        # print("Warning - non exact grid match")
        warnings.warn("Non exact grid match")
    else:
        raise ValueError("Grid mismatch")
    return ds_w.mean(dims)

def global_sum(ds,weights):
    """ Area weighted global mean of a DataArray"""
    # Check the shapes match
    assert ds.shape[-2:] == weights.shape, 'Shape mismatch in global_sum'
    assert ds.dims[-2:] == weights.dims, 'Coordinate mismatch in global_sum'
    dims = weights.dims
    if (weights[dims[0]]==ds[dims[0]]).all() and (weights[dims[1]]==ds[dims[1]]).all():
        ds_w = ds.weighted(weights)
    elif np.allclose(weights[dims[0]].data,ds[dims[0]].data) and np.allclose(weights[dims[1]].data,ds[dims[1]].data):
        ds_w = ds.weighted(weights.reindex_like(ds, method='nearest', tolerance=1e-4))
        # Warnings don't work properly?
        # print("Warning - non exact grid match")
        warnings.warn("Non exact grid match")
    else:
        raise ValueError("Grid mismatch")
    return ds_w.sum(dims)

def guess_lat_bounds(ds):
    """ Guess latitude bounds of dataset """
    try:
        lat = ds.lat
    except AttributeError:
        lat = ds.latitude
    assert lat.units == 'degrees_north'
    # Following iris guess_bounds
    points = np.empty(lat.shape[0] + 2)
    points[1:-1] = lat[:]
    points[0] = lat[0] - (lat[1]-lat[0])
    points[-1] = lat[-1] + (lat[-1]-lat[-2])
    diffs = np.diff(points)

    min_bounds = lat - diffs[:-1] * 0.5
    max_bounds = lat + diffs[1:] * 0.5
    bounds = np.array([min_bounds, max_bounds]).transpose()
    np.clip(bounds, -90, 90, out=bounds)
    return bounds

def guess_lon_bounds(ds):
    """ Guess longitude bounds of dataset """
    try:
        lon = ds.lon
    except AttributeError:
        lon = ds.longitude
    assert lon.units == 'degrees_east'
    # Following iris guess_bounds
    points = np.empty(lon.shape[0] + 2)
    points[1:-1] = lon[:]
    points[0] = lon[0] - (lon[1]-lon[0])
    points[-1] = lon[-1] + (lon[-1]-lon[-2])
    diffs = np.diff(points)

    min_bounds = lon - diffs[:-1] * 0.5
    max_bounds = lon + diffs[1:] * 0.5
    bounds = np.array([min_bounds, max_bounds]).transpose()
    return bounds

def get_area(ds):
    """ Calculate area DataArray from latitude and longitude bounds of dataset """
    try:
        lat_bnds = ds.lat_bnds
        # This is a workaround for FGOALS-g for which xarray gives a time dimension in
        # the bounds
        if 'time' in lat_bnds.dims:
            lat_bnds = lat_bnds.isel(time=0)
    except AttributeError:
        # For datasets w/o bounds
        lat_bnds = guess_lat_bounds(ds)
    try:
        lon_bnds = ds.lon_bnds
        if 'time' in lon_bnds.dims:
            lon_bnds = lon_bnds.isel(time=0)
    except AttributeError:
        # For datasets w/o bounds
        lon_bnds = guess_lon_bounds(ds)

    dlat = np.diff(np.sin(np.radians(lat_bnds)))
    dlon = np.diff(lon_bnds)
    A = np.outer(dlat, dlon)
    A /= A.sum()
    try:
        area = xr.DataArray(A, coords=[ds.lat.values, ds.lon.values], dims=['lat', 'lon'])
    except AttributeError:
        area = xr.DataArray(A, coords=[ds.latitude.values, ds.longitude.values], dims=['latitude', 'longitude'])
    return area

def remove_trend(ds,verbose=False):
    """ Calculate and remove a linear trend from a dataset """
    nt = len(ds)
    x = np.arange(nt)
    A = np.vstack([x, np.ones(len(x))]).T
    # ax + b
    a, b = np.linalg.lstsq(A, ds.data, rcond=None)[0]
    ds_new = ds.copy()
    ds_new.data -= a*np.arange(nt)
    if verbose:
        return ds_new, a, b
    else:
        return ds_new

def annual_cycle(ds):
    """ Mean annual cycle """
    month_length = ds.time.dt.days_in_month
    # Eventually use weighted ??? https://github.com/pydata/xarray/issues/3937
    weights = month_length.groupby('time.month') / month_length.groupby('time.month').sum()
    mean = (ds*weights).groupby('time.month').sum(dim='time')
    return mean

def rms_diff(ds1,ds2,area):
    """
    Global mean RMS difference

    Args:
        ds1 (xarray dataset)
        ds2 (xarray dataset)
        area (xarray dataset): Area
    """
    d = ds1 - ds2
    var = global_mean(d*d,area)
    return var**0.5

def seasonal_mean(ds):
    month_length = ds.time.dt.days_in_month
    # Use offset to get time near middle of season
    result = ((ds * month_length).resample(time='QS-DEC',loffset='45D').sum() /
          month_length.resample(time='QS-DEC',loffset='45D').sum())
    return result

def seasonal_cycle(ds):
    """ Mean seasonal cycle """
    month_length = ds.time.dt.days_in_month
    # Eventually use weighted ??? https://github.com/pydata/xarray/issues/3937
    weights = month_length.groupby('time.season') / month_length.groupby('time.season').sum()
    mean = (ds*weights).groupby('time.season').sum(dim='time')
    # At this stage mean is in alphabetical order DJF, JJA, MAM, SON
    # Add a numerical index and sort on that.
    mean = mean.assign_coords({"iseas": ("season", [0,2,1,3])}).sortby('iseas')
    return mean

def total_mean(ds):
    """ Properly month length weighted mean of a DataArray"""
    month_length = ds.time.dt.days_in_month
    mean = (ds*month_length).sum(dim='time') / month_length.sum()
    return mean
