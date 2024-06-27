from typing import List, Optional
from enum import Enum
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns

from lib import get_runs_dataframe

class DataType(Enum) :
    DURATION = 1
    BROKEN_WC_TIME = 2
    DURATION_ADJUSTED_FOR_WC = 3
    
    
class DisplayType(Enum) :
    VALUE = 1
    VARIATION = 2

    
class TimeSeries :
        def __init__(self, index, values, name, label = None, color = None, individual_labels = None) -> None:
            self.index = index
            self.values = values
            self.name = name
            self.label = label
            self.color = color
            self.individual_labels = individual_labels

class BoxPlotter:
    def __init__(self, use_trace_cmd_cache: bool, trace_cmd_cache_filename: str) -> None:
        self.time_series: List[TimeSeries] = []
        self.use_trace_cmd_cache = use_trace_cmd_cache
        self.trace_cmd_cache_filename = trace_cmd_cache_filename
    
    def add_time_series(self, name, values, index = None, label = None, color = None) :
        if index == None :
            index = [i for i in range(len(values))]
        self.time_series.append(TimeSeries(index, values, name = name, label = label, color = color))
        
    def add_trace_cmd_run(self, name, df: pd.DataFrame, data: DataType, label = None, color = None) :
        match data:
            case DataType.DURATION:
                series = df["duration"]
            case DataType.BROKEN_WC_TIME:
                series = df["broken_wc_time"]
            case DataType.DURATION_ADJUSTED_FOR_WC:
                series = df["duration"] - df["broken_wc_time"]
            case _:
                raise Exception("Invalid data argument passed")
        
        self.time_series.append(TimeSeries(
            df.index, series, name = name, label = label, color = color, individual_labels = df['run_index']
        ))
    
    def add_trace_cmd_directory(self, benchmark_dir_path, name_prefix = "", data = DataType.DURATION, sort_by = None) :
        # benchmark_dir_path = get_result_dir_path(TRACE_CMD_SUB_DIR + "/" + benchmark_dir_name)
        contents = os.listdir(benchmark_dir_path)
        for c in contents :
            dir_path = os.path.join(benchmark_dir_path, c)
            if not os.path.isdir(dir_path) :
                continue
            
            df = get_runs_dataframe(
                dir_path, 
                self.use_trace_cmd_cache, 
                self.trace_cmd_cache_filename, 
                include_broken_wc = (data != DataType.DURATION)
            )
            
            if df is None :
                continue
            if sort_by is not None :
                df.sort_values(sort_by, ignore_index=True, inplace=True)
            self.add_trace_cmd_run(name_prefix + " " + c, df, data, name_prefix + " " + data.name + " " + c, None)
            
    
    def print_filter_names(self) :
        print("Available time series names :", [ts.name for ts in self.time_series])
        
    def matches_filters(self, value, filters) :
        for f in filters :
            if f in value :
                return True
        return False
    
    def get_series_filtered(self, fuzzy_filters: Optional[List[str]], exact_filters: Optional[List[str]]) :
        if not exact_filters and not fuzzy_filters :
            return self.time_series
        if fuzzy_filters is None :
            fuzzy_filters = []
        exact_filters = set(exact_filters) if exact_filters is not None else set()
        
        def matches(value) :
            if value in exact_filters :
                return True
            if fuzzy_filters :
                for sf in fuzzy_filters :
                    if sf in value :
                        return True
            return False
        
        return [ts for ts in self.time_series if matches(ts.name)]
    
    def get_series_in_order(self, fuzzy_filters: Optional[List[str]]) :
        if fuzzy_filters is None :
            return self.time_series
            
        result = []
        for filter in fuzzy_filters :
            result += sorted([ts for ts in self.time_series if filter in ts.name], key=lambda x: x.name)
        return result
    
    def show_violin(self, displayType = DisplayType.VALUE, filters = None, exact_filters = None, size_factor = 2.5, title = None) :
        filtered_ts = self.get_series_filtered(fuzzy_filters=filters, exact_filters=exact_filters)
        # print([ts.name for ts in filtered_ts])
        
        # data_df = pd.DataFrame({f"{ts.name} ({len(ts.values)})": ts.values for ts in filtered_ts})
        data_df = pd.DataFrame({ts.name: ts.values for ts in filtered_ts})
        means = data_df.mean()
        sorted_data_df = data_df[means.sort_values().index]
        if displayType == DisplayType.VARIATION :
            reference_value = min(means)
            for col in sorted_data_df :
                sorted_data_df[col] = ((sorted_data_df[col] - reference_value) * 100) / reference_value
            
        melted_df = sorted_data_df.melt(var_name='experiment', value_name='value')
        melted_df['numa balancing'] = melted_df['experiment'].apply(lambda x : "nb-enabled" in x)
        melted_df['stripped'] = melted_df['experiment'].apply(lambda x : x.replace("nb-enabled-", "").replace("nb-disabled-", ""))
        sns.violinplot(data=melted_df, x='stripped', y='value', hue="numa balancing", split=True, gap=.1, density_norm='area', inner='quart')
            
        if title :
            plt.title(title)
        else :
            if displayType == DisplayType.VARIATION :
                plt.title(f"Runtime variation - reference runtime : {reference_value:.2f} s")
                plt.ylabel("% slower than best average runtime")
            else :
                plt.title(f"Runtime")
                plt.ylabel("Runtime")
        plt.xlabel("")
        # # sns.violinplot(data = sorted_data_df, scale="area", inner_kws=dict(box_width=15, whis_width=7, color=".3"))
        plt.gcf().set_size_inches(len(sorted_data_df.columns) * size_factor * 0.6, size_factor * 4)
        plt.xticks(rotation=20, ha='right')
        plt.minorticks_on()
        plt.grid(axis="y", which="major", color='0.2', linestyle='-', linewidth=0.5)
        plt.grid(axis="y", which="minor", color='0.1', linestyle=':', linewidth=0.5)
        
        plt.show()
            
            
    def show_variations(self, title = None, ordered_filters = None, exact_filters = None, w = 12, h = 6) :
        
        # filtered_ts = self.get_series_filtered(fuzzy_filters=filters, exact_filters=exact_filters)
        if ordered_filters is None :
            ordered_filters = ["sockets", "sockorder", "sequential", "none"]
        filtered_ts = self.get_series_in_order(ordered_filters)
        print([ts.name for ts in filtered_ts])
            
        # data_df = pd.DataFrame({f"{ts.name} ({len(ts.values)})": ts.values for ts in filtered_ts})
        data_df = pd.DataFrame({f"{ts.name} ({len(ts.values)})": ts.values for ts in filtered_ts})
        means = data_df.mean()
        reference_value = min(means)
        # sorted_data_df = data_df[means.sort_values().index]
        sorted_data_df = data_df
        for col in sorted_data_df :
            sorted_data_df[col] = ((sorted_data_df[col] - reference_value) * 100) / reference_value
        
        plt.gcf().set_size_inches(w, h)
        plt.ylabel("% slower than best average runtime")
        g = sns.boxplot(data = sorted_data_df, showmeans=True, meanprops={'marker':'D',
                       'markerfacecolor':'white', 
                       'markeredgecolor':'black',
                       'markersize':'5'})
        # g.set_yscale("log")
        plt.xticks(rotation=18, ha='right')
        plt.minorticks_on()
        plt.grid(axis="y", which="both")
        plt.grid(axis="x", which="major")
        plt.title(f"Runtime variation - reference runtime : {reference_value:.2f} s")
        plt.show()
        
    def show_times(self, name_filters = None) :
        plt.ylabel("Time (seconds)")
        filtered_ts: List[TimeSeries] = self.time_series
        if name_filters is not None and len(name_filters) > 0 :
            filtered_ts = [ts for ts in self.time_series if self.matches_filters(ts.name, name_filters)]
        
        data_df = pd.DataFrame({f"{ts.name} ({len(ts.values)})": ts.values for ts in filtered_ts}) # [ts.values for ts in filtered_ts]
        sorted_data_df = data_df[data_df.mean().sort_values().index]
        sns.boxplot(data = sorted_data_df)
        plt.xticks(rotation=20, ha='right')
        plt.minorticks_on()
        plt.grid(axis="y", which="both")
        plt.grid(axis="x", which="major")
        plt.legend(loc='best')
        plt.show()
        
    def show_times_adjusted_for_wc(self, name_filters = None, show_labels = False) :
        pass

    
    # def plot_trace_cmd_directory(self, benchmark_dir_name, label_prefix, data = "duration", sort_by = None, update_reference_value = True, folder_name_filters = None)

    
    def reset(self) :
        self.time_series = []