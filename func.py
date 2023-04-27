import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from matplotlib.widgets import SpanSelector
import xarray as xr
# %%


class span_select():

    def __init__(self, ax_in, canvas, orient='horizontal'):
        self.ax_in = ax_in
        self.canvas = canvas
        self.selector = SpanSelector(
            self.ax_in, self, orient, span_stays=True, useblit=True
        )

    def __call__(self, min, max):
        self.min = pd.Timestamp(min, unit='d')
        self.max = pd.Timestamp(max, unit='d')


def cloud_base(df):
    df = df.where(df.range < 4000, drop=True)
    file_date = np.datetime_as_string(df.time[0], 'D').replace('-', '')
    print(file_date)
    cloudbase = ((np.diff(df['depolarisation'], axis=1) > 0) *
                 (np.diff(df['beta'], axis=1) > 0) *
                 (df['beta'][:, :-1] > 5e-7))  # threshold for all the values in a cloud

    cloudbase = median_filter(cloudbase, size=(1, 7))

    c = np.where(np.concatenate((cloudbase[:, 0].reshape(-1, 1), cloudbase[:, :-1] != cloudbase[:, 1:],
                                 np.repeat(True, cloudbase.shape[0]).reshape(-1, 1)), axis=1))
    d = np.split(c[1], np.unique(c[0], return_index=True)[1][1:])

    # There is no gap in the increasing depo+beta and the first range gate must be larger than 150m
    # the Height will be like x[0]:x[1], meaning ignore the last x[1]
    e = np.array([[i, np.diff(x)[0], x[0], x[1]]
                 for i, x in enumerate(d) if (x.size == 3) & (x[0] > 30)])  # first height must be > 150m

    # cloud thickness before attenuation must be 50m to 150m and max height < 3500m
    # cloudbase_result = e[(e[:, 1] > 7) & (e[:, 1] < 30) & (e[:, 2] < 700)]
    if e.size < 1:
        print('no liquid clouds')
        return None
    cloudbase_result = e[(e[:, 1] < 30) & (e[:, 2] < 700)]

    if cloudbase_result.size < 1:
        print('no liquid clouds')
        return None
    result_ = []
    for i in cloudbase_result:
        i[2] = i[2] + np.argmin(df['depolarisation'].isel(time=i[0],
                                range=slice(i[2], i[3])).values)
        profile = df.isel(time=i[0], range=slice(i[2], i[2]+40))
        depo = profile['depolarisation'].values
        if depo[0] > 0.02:
            continue

        depth = np.arange(40)
        time = profile['time'].values
        range = profile['range'].values
        beta = profile['beta'].values
        n_no_att = i[3] + 1 - i[2]
        no_att = np.concatenate([np.repeat(True, n_no_att), np.repeat(False, 40 - n_no_att)])
        if beta[n_no_att] < 2e-5:  # beta at max must be a cloud
            continue
        result_.append(xr.Dataset(
            data_vars=dict(
                depo=(["time", "depth"], [depo]),
                range=(["time", "depth"], [range]),
                beta=(["time", "depth"], [beta]),
                no_att=(["time", "depth"], [no_att])
            ),
            coords=dict(
                time=[time],
                depth=(["depth"], depth)
            )))

    if len(result_) < 1:  # check if there is any liquid clouds
        print('no liquid clouds')
        return None
    result = xr.concat(result_, dim="time")

    temp = pd.DataFrame({'time': pd.Series(result['time']),
                         'range': pd.Series(result['range'].values[:, 0])})
    rolling20 = temp.set_index('time').rolling('20min', center=True)
    check_cluster = rolling20.count()
    check_stable_level = (rolling20.max() - rolling20.min())
    # more than 15 profiles every10mins, and fluctuation height less than 200m
    liquid_time = check_cluster[(check_cluster['range'] > 50) &
                                (check_stable_level['range'] < 200)].index

    # temp = pd.DataFrame({'time': pd.Series(result['time']),
    #                      'range': pd.Series(result['range'])})
    # check = temp.set_index('time').rolling('10min', center=True).count()
    # liquid_time = check[check['range'] > 15].index
    result = result.sel(time=liquid_time)
    if result.time.size < 1:
        print('no liquid clouds')
        return None
    return result


# %%
