import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
import matplotlib.dates as mdates
import func
from matplotlib.widgets import SpanSelector
from sklearn.linear_model import LinearRegression
myFmt = mdates.DateFormatter('%H:%M')

# %%
path = "/home/le/Data/Kenttarova"
files_radar = glob.glob(path + '/*rpg*.nc')
for file_ceilo in sorted(glob.glob(path + '/*cl61*.nc')):

    ceilo = xr.open_dataset(file_ceilo)
    ceilo = ceilo.where(ceilo.range < 4000, drop=True)
    file_date = file_ceilo.split('/')[-1].split('_')[0]
    file_radar = [x for x in files_radar if file_date in x]
    print(file_date)
    if len(file_radar) > 0:
        radar = xr.open_dataset(file_radar[0])
        radar = radar.where(radar.range < 4000, drop=True)
        radar_yes = True
        print('radar files available')
    else:
        radar_yes = False

    result = func.cloud_base(ceilo)
    if result is None:
        continue
    result.to_netcdf(path + '/result/' + file_date + '_cloudbase.nc')
    first = result.isel(depth=0)
    fig, ax = plt.subplots(2, 1, figsize=(16, 6), sharex=True)
    p = ax[0].pcolormesh(ceilo['time'].values, ceilo['range'], ceilo['beta'].T,
                         norm=LogNorm(vmin=1e-7, vmax=1e-4))
    colorbar = fig.colorbar(p, ax=ax[0])
    ax[0].plot(first['time'], first['range'], 'r.', markersize=0.5)
    # ax[0].set_ylim([0, 4000])

    p = ax[1].pcolormesh(ceilo['time'].values, ceilo['range'], ceilo['depolarisation'].T,
                         vmin=0, vmax=0.5)
    colorbar = fig.colorbar(p, ax=ax[1])
    ax[1].plot(first['time'], first['range'], 'r.', markersize=0.5)
    # ax[1].set_ylim([0, 4000])
    fig.savefig(path + '/Img/' + file_date + '_cloud.png',
                bbox_inches='tight', dpi=600)
    plt.close('all')

    fig, ax = plt.subplots(3, 2, figsize=(16, 12), constrained_layout=True)
    coef = []
    for i in result['time']:
        profile = result.sel(time=i)
        ax[1, 0].plot(profile['depo'], profile['depth'], alpha=0.5)
        ax[1, 1].plot(profile['beta'], profile['depth'], alpha=0.5)
        model = LinearRegression()
        profile_ = pd.DataFrame({'depo': profile['depo'].values,
                                 'depth': profile['depth'].values})
        profile_ = profile_.iloc[:20, :]
        profile_ = profile_.dropna(axis=0)
        model.fit(profile_['depth'].values.reshape(-1, 1), profile_['depo'].values)
        coef.append(model.coef_[0])
        ax[2, 0].plot(model.predict(profile_['depth'].values.reshape(-1, 1)),
                      profile_['depth'], alpha=0.5)

    ax[2, 1].plot(result['time'].values, coef, '.')
    if radar_yes:
        axtwin = ax[2, 1].twinx()
        axtwin.plot(radar['time'], radar['lwp'], 'r.')
        axtwin.set_ylabel('LWP', color='r')
        axtwin.grid()

    ax[1, 0].set_xlim([0, 0.2])
    ax[2, 0].set_xlim([0, 0.2])
    ax[1, 1].set_xscale('log')
    ax[2, 1].xaxis.set_major_formatter(myFmt)
    for ax_ in ax.flatten():
        ax_.grid()
    ax[1, 0].set_xlabel(r'$\delta$')
    ax[2, 0].set_xlabel('fitted $\delta$ to \n 20th range gates')
    ax[2, 1].set_ylabel('Slope (a) of \n $\delta$ = a*r + b', color='tab:blue')
    ax[2, 1].set_xlabel('Time')
    ax[2, 1].set_ylim([0.0003, 0.003])
    ax[1, 1].set_xlabel(r'$\beta$')
    for ax_ in ax.flatten()[:-1]:
        ax_.set_ylim([0, 40])
        ax_.set_ylabel('Depth [range gate]')

    H, depo_edges, range_edges = np.histogram2d(result['depo'].values.flatten(),
                                                np.tile(result['depth'], result['depo'].shape[0]),
                                                bins=[np.linspace(0, 0.2, 20), np.arange(40)])
    X, Y = np.meshgrid(depo_edges, range_edges)
    H[H < 1] = np.nan
    p = ax[0, 0].pcolormesh(X, Y, H.T)
    ax[0, 0].set_ylabel('Depth [range gate]')
    ax[0, 0].set_xlabel(r'$\delta$')
    colorbar = fig.colorbar(p, ax=ax[0, 0])
    colorbar.ax.set_ylabel('N')

    H, depo_edges, range_edges = np.histogram2d(np.log10(result['beta'].values).flatten(),
                                                np.tile(result['depth'], result['beta'].shape[0]),
                                                bins=[np.linspace(-7, -3, 20), np.arange(40)])
    X, Y = np.meshgrid(depo_edges, range_edges)
    H[H < 1] = np.nan
    p = ax[0, 1].pcolormesh(X, Y, H.T)
    ax[0, 1].set_ylabel('Depth [range gate]')
    ax[0, 1].set_xlabel(r'$\beta$')
    colorbar = fig.colorbar(p, ax=ax[0, 1])
    colorbar.ax.set_ylabel('N')
    fig.savefig(path + '/Img/' + file_date + '_cloudstat.png',
                bbox_inches='tight', dpi=600)
    plt.close('all')
