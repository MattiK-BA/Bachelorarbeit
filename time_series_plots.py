import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy import stats
import statistics
from functools import reduce
import seaborn as sns

plt.rcParams['text.usetex'] = True

stories = pickle.load(open("analyses/data/stories.p", "rb"))
stories.drop(stories[stories['stopped'] == True].index, inplace = True)

false_stories = stories.loc[stories.number_rating <= 2]
mixed_stories = stories.loc[stories.number_rating == 3]
na_stories = stories.loc[stories.number_rating.isna()]
true_stories = stories.loc[stories.number_rating >= 4]

def helper_time_resampling(time_points, freq = "1440min", truncate_to_n_datapoints = 365):
    time_points = pd.to_datetime(time_points)
    t_index = pd.DatetimeIndex(pd.date_range(start=min(time_points).floor(freq).to_pydatetime(), end=max(time_points).floor(freq).to_pydatetime(), freq=freq))
    resampled_time_series = pd.Series(np.ones(len(time_points)), index = time_points).resample(freq).sum().reindex(t_index).fillna(0)
    index_free_time_series = resampled_time_series.reset_index()[0]
    
    index_free_and_equal_len_time_series = index_free_time_series[0:truncate_to_n_datapoints]
    if(len(index_free_and_equal_len_time_series) < truncate_to_n_datapoints):
        index_free_and_equal_len_time_series = np.concatenate((np.array(index_free_time_series), np.zeros(truncate_to_n_datapoints-len(index_free_time_series))))
    else:
        index_free_and_equal_len_time_series = index_free_time_series
    
    return resampled_time_series, index_free_time_series, index_free_and_equal_len_time_series
    
time_series_stories = []
for idx, story in false_stories.iterrows():
    ts = helper_time_resampling(story['_time_series_query_tweets'], freq = "1440min", truncate_to_n_datapoints = 425)[2]
    ts_relative = np.array(ts / max(ts))
    ts_relative.sort()
    time_series_stories.append(ts_relative[::-1])
    
    #peak_adjusted_ts = ts[np.clip(np.argmax(ts)-7, 0, 365):len(ts)] 
    #plt.plot(peak_adjusted_ts / max(ts), color = "black", alpha = 1/len(false_stories))
    
#plt.show()
range_thres = (0.03, 0.5)
range_gap = (3, 21)

def helper_heatmap_data_prep(observed_data, range_thres, range_gap, relative_metric):
    threshold_levels = [round(x, 3) for x in list(np.linspace(range_thres[0], range_thres[1], int((range_thres[1]-range_thres[0])*1e2+1), endpoint=True))]
    gap_levels = np.concatenate((np.arange(range_gap[0], range_gap[1], 1), np.array([range_gap[1]])))

    n_peak_parameter_crossing = pd.DataFrame({}) 
    i_p = 0

    for thres in threshold_levels:
        for gap in gap_levels:
            peaks,_ = find_peaks(observed_data, distance=gap, height = thres*relative_metric)
            comb = pd.DataFrame({'thres': thres, 'gap': gap, 'peaks': len(peaks)}, index = [i_p])
            n_peak_parameter_crossing = pd.concat([n_peak_parameter_crossing, comb])
            
            i_p = i_p + 1
    
    return n_peak_parameter_crossing.pivot("thres", "gap", "peaks")


def helper_stories_iterator(stories_oi):
    i = 0
    heatmaps = []
    for idx, story in stories_oi.iterrows():
        print("analyzing story:" + str(i) + "; out of:" + str(len(stories_oi)))
        ts1, ts2, ts3 = helper_time_resampling(story['_time_series_query_tweets'], freq = "1440min", truncate_to_n_datapoints = 425)
        heatmaps.append(helper_heatmap_data_prep(ts1, range_thres, range_gap, max(ts1)))
        i = i +1
    return heatmaps

true_stories_heatmaps = helper_stories_iterator(true_stories)
true_stories_mean_heatmap = reduce(pd.DataFrame.add, true_stories_heatmaps) / len(true_stories_heatmaps)
true_stories_mean_heatmap_log = true_stories_mean_heatmap.applymap(np.log2)

false_stories_heatmaps = helper_stories_iterator(false_stories)
false_stories_mean_heatmap = reduce(pd.DataFrame.add, false_stories_heatmaps) / len(false_stories_heatmaps)
false_stories_mean_heatmap_log = false_stories_mean_heatmap.applymap(np.log2)

differences = false_stories_mean_heatmap-true_stories_mean_heatmap
differences_log = false_stories_mean_heatmap_log-true_stories_mean_heatmap_log


# raw data plot    
max_value = max(true_stories_mean_heatmap.values.max(), false_stories_mean_heatmap.values.max())
fig, axs = plt.subplots(1, 3, sharex = False, sharey = False, figsize=(7.1, 3.4))

h1 = sns.heatmap(true_stories_mean_heatmap[::-1], cmap="PuOr",cbar=True, cbar_ax=None, vmin = 1, vmax = max_value, ax=axs[0], xticklabels=3, yticklabels=10)
h2 = sns.heatmap(false_stories_mean_heatmap[::-1], cmap="PuOr", cbar = True, cbar_ax=None, vmin = 1, vmax = max_value, ax=axs[1], xticklabels=3, yticklabels=10)

limit = max(abs(differences.values.max()), abs(differences.values.min()))

h3 = sns.heatmap(differences[::-1], cmap="RdYlGn", cbar = True, ax=axs[2], vmin = -limit, vmax = limit, xticklabels=3, yticklabels=10)



axs[0].set_title("True stories")
axs[1].set_title("False stories")
axs[2].set_title("Difference")

axs[0].set_xlabel("Distance")
axs[0].set_ylabel("Height in percent of maximal peak")

axs[1].set_xlabel("Distance")
axs[1].set_ylabel("")
axs[2].set_xlabel("Distance")
axs[2].set_ylabel("")

plt.show()


# raw data plot    
max_value = max(true_stories_mean_heatmap_log.values.max(), false_stories_mean_heatmap_log.values.max())
fig, axs = plt.subplots(1, 3, sharex = False, sharey = False, figsize=(7.1, 3.4))

h1 = sns.heatmap(true_stories_mean_heatmap_log[::-1], cmap="PuOr",cbar=True, cbar_ax=None, vmin = np.log2(1), cbar_kws={'label': 'log2(n)'}, vmax = max_value, ax=axs[0], xticklabels=3, yticklabels=10)
h2 = sns.heatmap(false_stories_mean_heatmap_log[::-1], cmap="PuOr", cbar = True, cbar_ax=None, vmin = np.log2(1), cbar_kws={'label': 'log2(n)'}, vmax = max_value, ax=axs[1], xticklabels=3, yticklabels=10)

limit = max(abs(differences.values.max()), abs(differences.values.min()))

h3 = sns.heatmap(differences[::-1], cmap="RdYlGn", cbar = True, ax=axs[2], vmin = -limit, vmax = limit, cbar_kws={'label': 'Absolute Difference'}, xticklabels=3, yticklabels=10)



axs[0].set_title("True stories")
axs[1].set_title("False stories")
axs[2].set_title("Difference")

axs[0].set_xlabel("Distance")
axs[0].set_ylabel("Height in percent of maximal peak")

axs[1].set_xlabel("Distance")
axs[1].set_ylabel("")
axs[2].set_xlabel("Distance")
axs[2].set_ylabel("")

plt.show()



p_values_test = pd.DataFrame().reindex_like(true_stories_heatmaps[0])

for i_thres in p_values_test.index:
    for i_gap in p_values_test.columns:
        print(i_thres)
        dist1 = [story.loc[i_thres, i_gap] for story in false_stories_heatmaps]
        dist2 = [story.loc[i_thres, i_gap] for story in true_stories_heatmaps]
        p_values_test.loc[i_thres, i_gap] = stats.mannwhitneyu(dist1, dist2, alternative = "greater").pvalue

sns.heatmap(p_values_test[::-1], cbar = True)
plt.show()

#stats.kstest(peak_dist_true, peak_dist_false, alternative = 'greater')