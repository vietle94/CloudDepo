import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import LogNorm
from scipy.ndimage import median_filter
import glob

# %%
path = r'C:\Users\le\Desktop\Data\Kenttarova/'
file_paths = glob.glob(path + '/*.nc')
for file in file_paths:
    # df = xr.open_dataset(file_paths[2])
    df = xr.open_dataset(file)
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

    cloudbase_result = e[(e[:, 1] > 10) & (e[:, 1] < 30) & (e[:, 2] < 700)]
    result = pd.DataFrame({})
    for i, _ in enumerate(cloudbase_result):
        time = df['time'][cloudbase_result[i, 0]].values
        if df['depolarisation'][cloudbase_result[i, 0], cloudbase_result[i, 2]].values > 0.05:
            continue
        # if df['beta'][cloudbase_result[i, 0], cloudbase_result[i, 3]].values < 2e-5:
        #     continue
        depo = df['depolarisation'][cloudbase_result[i, 0],
                                    # cloudbase_result[0, 2]:cloudbase_result[0, 3]].values
                                    cloudbase_result[i, 2]:cloudbase_result[i, 2]+40].values
        beta = df['beta'][cloudbase_result[i, 0],
                          # cloudbase_result[0, 2]:cloudbase_result[0, 3]].values
                          cloudbase_result[i, 2]:cloudbase_result[i, 2]+40].values
        range = df['range'][cloudbase_result[i, 2]:cloudbase_result[i, 2]+40].values
        depth = range - range[0]
        n_no_att = cloudbase_result[i, 3] + 1 - cloudbase_result[i, 2]

        no_att = np.concatenate([np.repeat(True, n_no_att), np.repeat(False, 40 - n_no_att)])
        result_ = pd.DataFrame({'datetime': time,
                                'depo': depo,
                                'beta': np.log10(beta),
                                'range': range,
                                'depth': depth,
                                'no_att': no_att})
        result = result.append(result_, ignore_index=True)

    if result.shape[0] < 1:  # check if there is any liquid clouds
        continue

    result = result.set_index('datetime')
    result['cluster_check'] = result['depth'].resample('10min').transform('count') > 50*40
    result = result[result['cluster_check']].reset_index()
    result = result.drop('cluster_check', axis=1)
    if result.shape[0] < 1:  # check if there is any liquid clouds
        continue

    result.to_csv(path + file_date + '_clouddepo.csv', index=False)
    first = result.groupby('datetime').first().reset_index()
    fig, ax = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    p = ax[0].pcolormesh(df['time'].values, df['range'], df['beta'].T,
                         norm=LogNorm(vmin=1e-7, vmax=1e-4))
    colorbar = fig.colorbar(p, ax=ax[0])
    ax[0].plot(first['datetime'], first['range'], 'r.', markersize=0.5)
    ax[0].set_ylim([0, 4000])

    p = ax[1].pcolormesh(df['time'].values, df['range'], df['depolarisation'].T,
                         vmin=0, vmax=0.5)
    colorbar = fig.colorbar(p, ax=ax[1])
    ax[1].plot(first['datetime'], first['range'], 'r.', markersize=0.5)
    ax[1].set_ylim([0, 4000])
    fig.savefig(path + file_date + '_cloud.png', bbox_inches='tight', dpi=600,
                transparent=False)
    plt.close('all')

# %%
