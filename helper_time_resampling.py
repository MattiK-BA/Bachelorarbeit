import pandas as pd
import numpy as np
import datetime
import numpy as np

def helper_time_resampling(time_points, freq = "1440min", truncate_to_n_datapoints = 365, ts_start = None, ts_end = None):
    if (len(time_points) == 0):
       return None
    time_points = pd.to_datetime(time_points)
    if(ts_end != None and ts_start != None):
        t_index = pd.DatetimeIndex(pd.date_range(start=ts_start.floor(freq).to_pydatetime(), end=ts_end.floor(freq).to_pydatetime(), freq=freq))
    else:
        t_index = pd.DatetimeIndex(pd.date_range(start=min(time_points).floor(freq).to_pydatetime(), end=max(time_points).floor(freq).to_pydatetime(), freq=freq))
    resampled_time_series = pd.Series(np.ones(len(time_points)), index = time_points).resample(freq).sum().reindex(t_index).fillna(0)
    index_free_time_series = resampled_time_series.reset_index()[0]
    
    index_free_and_equal_len_time_series = index_free_time_series[0:truncate_to_n_datapoints]
    if(len(index_free_and_equal_len_time_series) < truncate_to_n_datapoints):
        index_free_and_equal_len_time_series = np.concatenate((np.array(index_free_time_series), np.zeros(truncate_to_n_datapoints-len(index_free_time_series))))
    else:
        index_free_and_equal_len_time_series = index_free_time_series
    
    return resampled_time_series, index_free_time_series, index_free_and_equal_len_time_series

