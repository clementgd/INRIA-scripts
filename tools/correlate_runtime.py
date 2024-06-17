#!/root/jupyter/python_env/bin/python3

import argparse
import os
import pandas as pd
import re
import json
from collections import defaultdict
from typing import Optional, Tuple, Dict, List
from experiments_lib import init_command_chain_with_config_thp, generate_perf_stat_batch, \
    get_sockorder_omp_places, get_sequential_omp_places, \
    get_perf_stat_cmd, run_shell_command, grid_experiment
    
import logging

# Set up basic configuration for logging
logging.basicConfig(
    level=logging.INFO,
    format='\n### PY ### %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'  # Only display hours, minutes, and seconds
)
    

EXTENSION = ".txt"
RESULTS_DIR_PATH = "/tmp/perf_stat"

# TODO Make the number of runs and warmup adjustable


# Run the program in each of the 6 configuration into /tmp/... and then use these as an input for correlation computation.
# Could also draw a diagram of the most correlated stuff ?
# But the best would be to have a graph of the most correlated features, in blue if positive, red if negative


    
def test_events(program_path: str, events: str):
    perf_stat_cmd = get_perf_stat_cmd(program_path, events, "./test_events_result.txt")
    print(f"Shell command : {perf_stat_cmd}")
    run_shell_command(perf_stat_cmd)
    
    
    
    
def collect_samples(program_path: str, events: str, nruns: int, nwarmups: int):
    def run_with_params(name: str, nb: bool, omp_places: Optional[str], thp_enabled: str, thp_defrag: str):
        logging.info(f"Running with params : name = {name}, nb = {nb}, ompPlaces = {omp_places}, thp_enabled = {thp_enabled}, thp_defrag = {thp_defrag}")
        command_chain = init_command_chain_with_config_thp(nb, omp_places, thp_enabled, thp_defrag)
        command_chain += generate_perf_stat_batch(program_path, nruns, nwarmups, RESULTS_DIR_PATH, name, events)
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
    
    
    
    
    
    
def parse_per_cpu_result_file(file_path) -> Tuple[pd.DataFrame, Dict]:
    with open(file_path, 'r') as file:
        text = file.read()
    nas_time_match = re.search(r'Time in seconds\s+=\s+(\d*.\d*)', text)
    if nas_time_match :
        nas_time = float(nas_time_match.group(1))
        
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
    
    events_df["ml_l3_mr.all"] = events_df["mem_load_l3_miss_retired.remote_dram"] + events_df["mem_load_l3_miss_retired.local_dram"]
    # events_df["mem_load_l3_miss_retired-over-LLC-all"] = events_df["ml_l3_mr.all"] / events_df["LLC-all-misses"]
    events_df["ml_l3_mr.remote_over_local_dram"] = events_df["mem_load_l3_miss_retired.remote_dram"] / events_df["mem_load_l3_miss_retired.local_dram"]
    events_df["ml_l3_mr.remote_over_total"] = events_df["mem_load_l3_miss_retired.remote_dram"] / events_df["ml_l3_mr.all"]
    events_df["ml_l3_mr.local_over_total"] = events_df["mem_load_l3_miss_retired.local_dram"] / events_df["ml_l3_mr.all"]
    
    # mem_load_l3_miss_retired.remote_fwd
    # mem_load_l3_miss_retired.remote_hitms
    events_df["ml_l3_mr.fwd_over_total"] = events_df["mem_load_l3_miss_retired.remote_fwd"] / events_df["ml_l3_mr.all"]
    events_df["ml_l3_mr.hitm_over_total"] = events_df["mem_load_l3_miss_retired.remote_hitm"] / events_df["ml_l3_mr.all"]
    return events_df, meta_values
    

# TODO Add posibility to specify events ?
def parse_batch_results(dir_path) :
    file_paths = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(EXTENSION)]
    concatenated_dict = defaultdict(list)
    for file_path in file_paths :
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
def parse_batches_results_from_benchmark(benchmark_dir_path: str) -> List[pd.DataFrame] :
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

    
    
    
    
def analyze_samples():
    pass






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", help="The program to run and analyze")
    parser.add_argument('--result_dir', help='The directory that contains a run to be analyzed')
    parser.add_argument
    args = parser.parse_args()
    target: str = args.target
    benchmark_dir = args.result_dir
    # if not os.path.isfile(target):
    #     logging.error(f"Unable to find target {target}")
    #     exit()
    print(benchmark_dir)
        
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
    logging.info(joined_events)
    logging.info(all_events)
    
    # collect_samples(target, all_events, 5, 1)
        
    # test_events(target, all_events)

        
        
