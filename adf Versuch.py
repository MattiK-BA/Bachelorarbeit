import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from helper_time_resampling import helper_time_resampling
from scipy.signal import find_peaks

from hpn import hpn
from set_tex_variable import LaTeX_variable_writer
tex_writer = LaTeX_variable_writer("g_main.tex")


matti_data = pd.read_pickle(open("matti_data.p", "rb"))
granger_results = []

def analyze_story(story_data):
    contra_woi = story_data['_contra_tweets']['created_at'].loc[(story_data['_contra_tweets']['created_at'] >= story_data['_contra_tweets']['created_at'].quantile(0.00)) & (story_data['_contra_tweets']['created_at'] <= story_data['_contra_tweets']['created_at'].quantile(0.95))]
    non_contra_woi = story_data['_non_contra_tweets']['created_at'].loc[(story_data['_non_contra_tweets']['created_at'] >= story_data['_non_contra_tweets']['created_at'].quantile(0.00)) & (story_data['_non_contra_tweets']['created_at'] <= story_data['_non_contra_tweets']['created_at'].quantile(0.95))]

    if len(contra_woi) == 0 or len(non_contra_woi) == 0:
        return None

    ts_start = min(min(contra_woi), min(non_contra_woi)) 
    ts_end = max(max(contra_woi), max(non_contra_woi)) 

    contra_resampled = helper_time_resampling(time_points=contra_woi, ts_start = ts_start, ts_end = ts_end, freq="30min")[0]
    non_contra_resampled = helper_time_resampling(time_points=non_contra_woi, ts_start = ts_start, ts_end = ts_end, freq="30min")[0]

    window_size =3
    contra_resampled_smooth = contra_resampled.rolling(window=window_size, min_periods=1, center=True).mean()
    non_contra_resampled_smooth = non_contra_resampled.rolling(window=window_size, min_periods=1, center=True).mean()
    
    onset_window = np.argmax(contra_resampled) -2
    end_window = onset_window + 48

    if(end_window > len(contra_resampled) or onset_window < 0):
        return None

    data_window_non_contra = non_contra_resampled_smooth.iloc[onset_window:end_window]
    data_window_contra = contra_resampled_smooth.iloc[onset_window:end_window]
    
    non_contra_peaks_within_window, _ = find_peaks(data_window_non_contra)
    Anzahl_peaks = len(non_contra_peaks_within_window)
    
    # Granger-Causality Test
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.tsa.stattools import adfuller
   
    _, pvalue_contra_smooth, _, _, _, _ = adfuller(contra_resampled_smooth)
    _, pvalue_non_contra_smooth, _, _, _, _ = adfuller(non_contra_resampled_smooth)
    
    from statsmodels.tsa.api import VAR
    Zusammen = pd.concat([contra_resampled_smooth,non_contra_resampled_smooth],axis='columns')
    model = VAR(Zusammen)
    for i in [1,2,3,4,5]:
        Ergebnisprüfung = model.fit(i)
        print('Lag Order =', i)
        print('AIC : ', Ergebnisprüfung.aic)
        print('BIC : ', Ergebnisprüfung.bic)
        print('FPE : ', Ergebnisprüfung.fpe)
        print('HQIC: ',Ergebnisprüfung.hqic, '\n')

    def adf_test(df):
        adf_result = adfuller(df.values)
        print('ADF Statistics: %f' % adf_result[0])
        print('p-value: %f' % adf_result[1])
        print('Critical values:')
        for key, value in adf_result[4].items():
            print('\t%s: %.3f' % (key, value))
    print(f'ADF Test:Story {i_story + 1}')
    adf_test(contra_resampled_smooth)
    adf_test(non_contra_resampled_smooth)

    from statsmodels.tsa.stattools import kpss

    def kpss_test(df):    
        statistic, p_value, n_lags, critical_values = kpss(df.values)
    
        print(f'KPSS Statistic: {statistic}')
        print(f'p-value: {p_value}')
        print(f'num lags: {n_lags}')
        print('Critial Values:')
        for key, value in critical_values.items():
            print(f'   {key} : {value}')
    print(f'KPSS Test:Story {i_story + 1}')
    kpss_test(contra_resampled_smooth)
    kpss_test(non_contra_resampled_smooth)

    
    # Stationary Test
    if pvalue_contra_smooth > 0.05 or pvalue_non_contra_smooth > 0.05:
        print("Keine Stationarität")
    else:
        if np.unique(data_window_contra).size == 1 or np.unique(data_window_non_contra).size == 1:
            print("Granger Test nicht möglich wegen Konstanten in den Daten.")
        else:
            gc_res = grangercausalitytests(np.column_stack((data_window_contra, data_window_non_contra)), maxlag=5)
            print(gc_res[1][0]['ssr_ftest'])
    
    ret_dict = {}
    ret_dict['Daten_contra'] = data_window_contra
    ret_dict['Daten_non_contra'] = data_window_non_contra
    ret_dict['Peaks innerhalb des Fensters'] = Anzahl_peaks
    ret_dict['Daten_contra_gesamt'] = contra_resampled
    ret_dict['Daten_non_contra_gesamt'] = non_contra_resampled
    return ret_dict 

print("Peak within Window of Opportunity:")

for i_story in range(len(matti_data)):
    ret_array = analyze_story(matti_data.iloc[i_story])
    if ret_array is None:
        continue
    contradaten = ret_array['Daten_contra']
    noncontradaten = ret_array['Daten_non_contra']
    plt.plot(noncontradaten.index, noncontradaten.values, label= 'non-contra')
    plt.plot(contradaten.index, contradaten.values, label='contra')
    plt.title(f'Story {i_story + 1}')
    
    plt.xlabel('Zeit')
    plt.xticks(rotation = 45) 
    plt.ylabel('Rollierender Mittelwert der Tweets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Story_{i_story+1}.png', dpi=300)
    plt.close() 
       
for i_story in range(len(matti_data)):
    ret_array = analyze_story(matti_data.iloc[i_story])
    if ret_array is None:
        continue
    contradaten_ganz = ret_array['Daten_contra_gesamt']
    noncontradaten_ganz = ret_array['Daten_non_contra_gesamt']
    plt.plot(noncontradaten_ganz.index, noncontradaten_ganz.values, label= 'non-contra')
    plt.plot(contradaten_ganz.index, contradaten_ganz.values, label='contra')
    plt.title(f'Story {i_story + 1}')
    plt.xlabel('Zeit')
    plt.xticks(rotation = 45)
    plt.ylabel('Rollierender Mittelwert der Tweets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Story_{i_story+1}_Gesamtverlauf.png', dpi=300)
    plt.close() 

for i_story in range(len(matti_data)):
    result, results = analyze_story(matti_data.iloc[i_story])
    if result is not None:
        granger_results.append(results)

print("Granger Causality Test results:", granger_results)
 