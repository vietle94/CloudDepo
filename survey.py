import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
import matplotlib.dates as mdates
import func
from matplotlib.widgets import SpanSelector

myFmt = mdates.DateFormatter('%H:%M')
%matplotlib qt

# %%
path = r'C:\Users\le\Desktop\Data\Hyytiala/'
# file_paths = glob.glob(path + '/*.nc')
ceilo = xr.open_dataset(glob.glob(path + 'Ceilometer/*.nc')[0])
radar = xr.open_dataset(glob.glob(path + 'Radar/*.nc')[0])
file_date = np.datetime_as_string(ceilo.time[0], 'D').replace('-', '')

# %%
fig, ax = plt.subplots(2, 2, figsize=(16, 6), constrained_layout=True, sharex=True)
p = ax[0, 0].pcolormesh(ceilo['time'].values, ceilo['range'].values, ceilo['beta'].T.values,
                        norm=LogNorm(vmin=1e-7, vmax=1e-4))
colorbar = fig.colorbar(p, ax=ax[0, 0])
colorbar.ax.set_ylabel('Beta [sr-1 m-1]')

p = ax[1, 0].pcolormesh(ceilo['time'].values, ceilo['range'].values, ceilo['depolarisation'].T.values,
                        vmin=0, vmax=0.5)
colorbar = fig.colorbar(p, ax=ax[1, 0])
colorbar.ax.set_ylabel('Depo')

p = ax[0, 1].pcolormesh(radar['time'].values, radar['range'].values, radar['Zh'].T.values,
                        vmin=-40, vmax=10)
colorbar = fig.colorbar(p, ax=ax[0, 1])
colorbar.ax.set_ylabel('Reflectivity [dbZ]')

# p = ax[1, 1].pcolormesh(radar['time'].values, radar['range'].values, radar['v'].T.values,
#                         vmin=-4, vmax=4)
# colorbar = fig.colorbar(p, ax=ax[1, 1])
# colorbar.ax.set_ylabel('Doppler velocity [m/s]')

ax[1, 1].plot(radar['time'], radar['lwp'])
ax[1, 1].set_ylabel('LWP [g m-2]')
ax[1, 1].grid()
ax[0, 0].set_ylim([0, 4000])
ax[0, 1].set_ylim([0, 4000])
ax[1, 0].set_ylim([0, 4000])

for ax_ in ax.flatten():
    ax_.xaxis.set_major_formatter(myFmt)
fig.savefig(path + file_date + '_preview.png', bbox_inches='tight',
            dpi=600)

# %%
result = func.cloud_base(ceilo)

# %%
first = result.groupby('datetime').first().reset_index()
fig, ax = plt.subplots(2, 2, figsize=(16, 6), constrained_layout=True, sharex=True)
p = ax[0, 0].pcolormesh(ceilo['time'].values, ceilo['range'].values, ceilo['beta'].T.values,
                        norm=LogNorm(vmin=1e-7, vmax=1e-4))
colorbar = fig.colorbar(p, ax=ax[0, 0])
colorbar.ax.set_ylabel('Beta [sr-1 m-1]')
ax[0, 0].plot(first['datetime'], first['range'], 'r.', markersize=0.5)


p = ax[1, 0].pcolormesh(ceilo['time'].values, ceilo['range'].values, ceilo['depolarisation'].T.values,
                        vmin=0, vmax=0.5)
colorbar = fig.colorbar(p, ax=ax[1, 0])
colorbar.ax.set_ylabel('Depo')
ax[1, 0].plot(first['datetime'], first['range'], 'r.', markersize=0.5)

p = ax[0, 1].pcolormesh(radar['time'].values, radar['range'].values, radar['Zh'].T.values,
                        vmin=-40, vmax=10)
colorbar = fig.colorbar(p, ax=ax[0, 1])
colorbar.ax.set_ylabel('Reflectivity [dbZ]')
ax[0, 1].plot(first['datetime'], first['range'], 'r.', markersize=0.5)

# p = ax[1, 1].pcolormesh(radar['time'].values, radar['range'].values, radar['v'].T.values,
#                         vmin=-4, vmax=4)
# colorbar = fig.colorbar(p, ax=ax[1, 1])
# colorbar.ax.set_ylabel('Doppler velocity [m/s]')

ax[1, 1].plot(radar['time'], radar['lwp'])
ax[1, 1].set_ylabel('LWP [g m-2]')
ax[1, 1].grid()
ax[0, 0].set_ylim([0, 4000])
ax[0, 1].set_ylim([0, 4000])
ax[1, 0].set_ylim([0, 4000])

for ax_ in ax.flatten():
    ax_.xaxis.set_major_formatter(myFmt)
fig.savefig(path + file_date + '_cloudbase.png', bbox_inches='tight',
            dpi=600)

# %%


class area_select():

    def __init__(self, x, y, z, ax_in, fig):
        self.x, self.y, self.z = x, y, z
        self.ax_in = ax_in
        self.canvas = fig.canvas
        self.fig = fig
        self.selector = RectangleSelector(
            self.ax_in,
            self,
            useblit=True,  # Process much faster,
            interactive=True  # Keep the drawn box on screen
        )

    def __call__(self, event1, event2):
        self.mask = self.inside(event1, event2)
        self.area = self.z[self.mask]
        self.range = self.y[self.maskrange]
        self.time = self.x[self.masktime]
        print(f'Chosen {len(self.area.flatten())} values')

    def inside(self, event1, event2):
        """
        Returns a boolean mask of the points inside the rectangle defined by
        event1 and event2
        """
        self.xcord = [event1.xdata, event2.xdata]
        self.ycord = [event1.ydata, event2.ydata]
        x0, x1 = sorted(self.xcord)
        y0, y1 = sorted(self.ycord)
        self.masktime = (self.x > x0) & (self.x < x1)  # remove bracket ()
        self.maskrange = (self.y > y0) & (self.y < y1)
        return np.ix_(self.maskrange, self.masktime)

# %%


class span_select():

    def __init__(self, x, y, ax_in, canvas, orient='horizontal'):
        self.x, self.y = x, y
        self.ax_in = ax_in
        self.canvas = canvas
        self.selector = SpanSelector(
            self.ax_in, self, orient, span_stays=True, useblit=True
        )

    def __call__(self, min, max):
        self.min = pd.Timestamp(min, unit='d')
        self.max = pd.Timestamp(max, unit='d')
        print(self.min, self.max)
        # self.maskx = (self.x > min) & (self.x < max)
        # self.selected_x = self.x[self.maskx]
        # self.selected_y = self.y[self.maskx]


fig, ax = plt.subplots()
ax.plot(radar['time'].values, radar['lwp'].values)
ax.xaxis.set_major_formatter(myFmt)

span = span_select(radar['time'].values, radar['lwp'].values, ax, fig)
# dir(span)
# span.x
# pd.to_datetime('19402.052014926074')
# span

# %%
fig, ax = plt.subplots(figsize=(12, 3))

ceilo_ = ceilo.where(ceilo.range < 4000, drop=True)

p = ax.pcolormesh(ceilo_['time'].values, ceilo_['range'].values, ceilo_['depolarisation'].T.values,
                  vmin=0, vmax=0.5)
colorbar = fig.colorbar(p, ax=ax)
colorbar.ax.set_ylabel('Depo')
span = span_select(ceilo_['time'].values, ceilo_['range'].values, ax, fig)
