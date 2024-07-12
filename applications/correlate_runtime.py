#!/root/jupyter/python_env/bin/python3

import argparse
import os
import pandas as pd
import re
import json
from collections import defaultdict
from typing import Optional, Tuple, Dict, List
from experiments_lib import init_command_chain_with_config_thp, init_command_chain_with_config, generate_perf_stat_batch, \
    get_sockorder_omp_places, get_sequential_omp_places, \
    get_perf_stat_cmd, run_shell_command, grid_experiment, get_benchmark_dir
    
import logging
    
def collect_samples_thp(program_path: str, events: str, results_dir_path: str, nruns: int, nwarmups: int):
    def run_with_params(name: str, nb: bool, omp_places: Optional[str], thp_enabled: str, thp_defrag: str):
        logging.info(f"Running with params : name = {name}, nb = {nb}, ompPlaces = {omp_places}, thp_enabled = {thp_enabled}, thp_defrag = {thp_defrag}")
        command_chain = init_command_chain_with_config_thp(nb, omp_places, thp_enabled, thp_defrag)
        command_chain += generate_perf_stat_batch(program_path, nruns, nwarmups, results_dir_path, name, events)
        command_chain.execute()
        
    grid_experiment(
        ["nb", "omp_places", "thp_enabled", "thp_defrag"],
        [
            [("on", True), ("off", False)],
            [("sequential", get_sequential_omp_places()), ("sockorder", get_sockorder_omp_places()), ("none", None)],
            [("never", "never"), ("always", "always")],
            [("never", "never"), ("always", "always")],
        ],
        run_with_params
    )
    
    return get_benchmark_dir(results_dir_path, program_path)
    

# TODO Rename with something like "collect perf stats ?"
def collect_samples(program_path: str, events: str, results_dir_path: str, nruns: int, nwarmups: int):
    def run_with_params(name: str, nb: bool, omp_places: Optional[str]):
        logging.info(f"Running with params : name = {name}, nb = {nb}, ompPlaces = {omp_places}")
        command_chain = init_command_chain_with_config(nb, omp_places)
        command_chain += generate_perf_stat_batch(program_path, nruns, nwarmups, results_dir_path, name, events)
        command_chain.execute()
        
    grid_experiment(
        ["nb", "omp_places"],
        [
            [("on", True), ("off", False)],
            [("sequential", get_sequential_omp_places()), ("sockorder", get_sockorder_omp_places()), ("none", None)]
        ],
        run_with_params
    )
    
    return get_benchmark_dir(results_dir_path, program_path)
    
    
    
    
    
    
def parse_per_cpu_result_file(file_path) -> Tuple[pd.DataFrame, Dict]:
    with open(file_path, 'r') as file:
        text = file.read()
    nas_time_match = re.search(r'Time in seconds\s+=\s+(\d*.\d*)', text)
    if nas_time_match :
        nas_time = float(nas_time_match.group(1))
    else:
        logging.error(f"Unable to find NAS time in file {file_path}")
        exit()
        
    json_begin_pos = text.find('{')
    json_end_pos = text.rfind('}')
    json_text = text[json_begin_pos:json_end_pos + 1]
    json_lines = json_text.split('\n')
    
    events_df = pd.DataFrame()
    
    meta_values = {"nas_runtime": nas_time}
    
    event_counters: Dict[str, list] = defaultdict(list)
    for line in json_lines :
        json_object = json.loads(line)
        event_str = json_object["event"].strip()
        if event_str in ["duration_time", "user_time", "system_time"] :
            meta_values[event_str] = float(json_object["counter-value"])
        else :
            event_counters[event_str].append((int(json_object["cpu"]), float(json_object["counter-value"])))
    
    events_df = pd.DataFrame({event: [e[1] for e in sorted(values)] for event, values in event_counters.items()})
    if "LLC-load-misses" in events_df and "LLC-store-misses" in events_df and "LLC-loads" in events_df and "LLC-stores" in events_df:
        events_df["LLC-all-misses"] = events_df["LLC-load-misses"] + events_df["LLC-store-misses"]
        events_df["LLC-load-misses-ratio"] = events_df["LLC-load-misses"] / events_df["LLC-loads"]
        events_df["LLC-store-misses-ratio"] = events_df["LLC-store-misses"] / events_df["LLC-stores"]
    
    # ml_l3_mr = Mem Load L3 Miss Remote
    events_df["ml_l3_mr.all"] = events_df["mem_load_l3_miss_retired.remote_dram"] + events_df["mem_load_l3_miss_retired.local_dram"]
    events_df["ml_l3_mr.remote_over_local_dram"] = events_df["mem_load_l3_miss_retired.remote_dram"] / events_df["mem_load_l3_miss_retired.local_dram"]
    events_df["ml_l3_mr.remote_over_total"] = events_df["mem_load_l3_miss_retired.remote_dram"] / events_df["ml_l3_mr.all"]
    events_df["ml_l3_mr.local_over_total"] = events_df["mem_load_l3_miss_retired.local_dram"] / events_df["ml_l3_mr.all"]
    events_df["ml_l3_mr.fwd_over_total"] = events_df["mem_load_l3_miss_retired.remote_fwd"] / events_df["ml_l3_mr.all"]
    events_df["ml_l3_mr.hitm_over_total"] = events_df["mem_load_l3_miss_retired.remote_hitm"] / events_df["ml_l3_mr.all"]
    return events_df, meta_values
    

# TODO Add posibility to specify events ?
def parse_batch_results(dir_path) :
    file_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(EXTENSION)]
    concatenated_dict = defaultdict(list)
    for file_path in file_paths :
        if "sample_measure_command" in os.path.basename(file_path):
            continue
        events_df, meta_values = parse_per_cpu_result_file(file_path)
        for series_name, series in events_df.items() :
            concatenated_dict[f"min:{series_name}"].append(series.min())
            concatenated_dict[f"max:{series_name}"].append(series.max())
            concatenated_dict[f"sum:{series_name}"].append(series.sum())
            concatenated_dict[f"avg:{series_name}"].append(series.sum() / len(series))
        for key, val in meta_values.items() :
            concatenated_dict[key].append(val)
    return pd.DataFrame(concatenated_dict)



# TODO Maybe return the number of CPUs so that we can compute the average when we have only the sum ?
def parse_batches_results_from_benchmark(benchmark_dir_path: str) -> Dict[str, pd.DataFrame] :
    dfs = {}
    contents = os.listdir(benchmark_dir_path)
    for c in contents :
        dir_path = os.path.join(benchmark_dir_path, c)
        if not os.path.isdir(dir_path) :
            continue
        print(f"Parsing batch in directory {dir_path}")
        dfs[c] = parse_batch_results(dir_path)
    return dfs


def combine_benchmark_dir(benchmark_dir_path: str) :
    df = pd.DataFrame()
    contents = os.listdir(benchmark_dir_path)
    for c in contents :
        dir_path = os.path.join(benchmark_dir_path, c)
        if not os.path.isdir(dir_path) :
            continue
        
        df = pd.concat([df, parse_batch_results(dir_path)], ignore_index=True)
    return df




def print_correlations(df: pd.DataFrame, target_column = "nas_runtime"):
    # We first remove all the avg columns if any because will give the same result as sum
    
    avg_columns = [col for col in df.columns if "avg:" in col]
    df = df.drop(avg_columns, axis=1)
    pd.set_option('display.max_rows', 500)
    correlation_df = df.corr()
    correlation_series = correlation_df[target_column].drop(target_column).sort_values(ascending=False, key=abs)

    threshold = 0.98
    previous = []
    for col in correlation_series.index:
        was_dropped = False
        for previous_col in previous:
            # print(correlation_df.loc[col, previous_col])
            if correlation_df.loc[col, previous_col] > threshold:
                # print(f"Found correlation > {threshold} between {col} and {previous_col}")
                correlation_series.drop(col, inplace=True)
                was_dropped = True
                break
        if not was_dropped and "time" not in col:
            previous.append(col)
            
    print(correlation_series)

    
def analyze_samples(benchmark_dir_path):
    dfs = parse_batches_results_from_benchmark(benchmark_dir_path)
    concatenated = pd.concat(dfs.values())
    print_correlations(concatenated)



if __name__ == "__main__":
    # Set up basic configuration for logging
    logging.basicConfig(
        level=logging.INFO,
        format='\n### PY ### %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'  # Only display hours, minutes, and seconds
    )
        
    EXTENSION = ".txt"
    RESULTS_DIR_PATH = "/tmp/perf_stat"
    
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", metavar="PROGRAM_PATH", help="The program to run and analyze")
    group.add_argument('--analyze', metavar="BENCHMARK_DIRECTORY", help='The directory containing a run to be analyzed')
    parser.add_argument('--nruns', help='Number of runs to perform for each configuration', type=int, default=10)
    parser.add_argument('--nwarmups', help='Number of warmups to perform for each configuration', type=int, default=1)
    args = parser.parse_args()
    
    if args.analyze is not None :
        if not os.path.isdir(args.analyze):
            logging.error(f"{args.analyze} is not a directory")
            exit()
        benchmark_dir = args.analyze
        analyze_samples(benchmark_dir)
        exit()
        
    if not os.path.isfile(args.run):
        logging.error(f"Unable to find target {args.run}")
        exit()
        
    events_per_category = {}
    events_per_category["l3_cache"] = [
        "LLC-loads",
        "LLC-load-misses",
        "LLC-stores",
        "LLC-store-misses",
        "cycle_activity.stalls_l3_miss"
    ]
    events_per_category["l3_cache_miss"] = [
        "mem_load_l3_miss_retired.local_dram",
        "mem_load_l3_miss_retired.remote_dram",
        "mem_load_l3_miss_retired.remote_fwd",
        "mem_load_l3_miss_retired.remote_hitm"
    ]
    events_per_category["l2_cache"] = [
        "l2_rqsts.all_demand_references",
        "l2_rqsts.all_demand_miss",
        "l2_rqsts.all_demand_data_rd",
        "l2_rqsts.demand_data_rd_miss"
    ]
    events_per_category["generic"] = [
        "cache-references",
        "cache-misses",
        "migrations",
        "context-switches"
    ]
    
    joined_events = [",".join(event_list) for event_list in events_per_category.values()]
    all_events = ",".join(joined_events)
    program_path = args.run
    benchmark_dir = collect_samples(program_path, all_events, RESULTS_DIR_PATH, args.nruns, args.nwarmups)
    analyze_samples(benchmark_dir)
