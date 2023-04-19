import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from matplotlib.widgets import SpanSelector

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

    cloudbase_result = e[(e[:, 1] > 10) & (e[:, 1] < 30) & (e[:, 2] < 700)]
    result = pd.DataFrame({})
    for i, _ in enumerate(cloudbase_result):
        time = df['time'][cloudbase_result[i, 0]].values
        cloudbase_result[i, 2] = cloudbase_result[i, 2] + np.argmin(
            df['depolarisation'][cloudbase_result[i, 0], cloudbase_result[i, 2]:cloudbase_result[i, 3]].values)
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
        print('no liquid clouds')
        return 'no liquid clouds'

    result = result.set_index('datetime')
    result['cluster_check'] = result['depth'].resample('10min').transform('count') > 50*40
    result = result[result['cluster_check']].reset_index()
    result = result.drop('cluster_check', axis=1)
    if result.shape[0] < 1:  # check if there is any liquid clouds
        print('no liquid clouds')
        return 'no liquid clouds'

    return result


# %%
