import pandas as pd
import subprocess
import re
import os


def get_run_index(filename: str) -> int :
    match = re.search(r'_(\d+)\.dat', filename)
    if match:
        return int(match.group(1))
    return None


def get_trace_duration(dat_file_path: str) -> float :
    print("Computing trace duration for file :", dat_file_path)
    
    result = subprocess.run(
        ["trace-cmd", "report", "--first-event", "--last-event", dat_file_path],
        stdout = subprocess.PIPE,
        universal_newlines = True
    )

    firsts = []
    lasts = []
    for line in result.stdout.split("\n"):
        match = re.search(r'\s+\d+\tFirst event:\s*(\d+\.\d+)\tLast event:\s*(\d+\.\d+)', line)
        if match:
            firsts.append(float(match.group(1)))
            lasts.append(float(match.group(2)))
    if len(firsts) == 0 or len(lasts) == 0 :
        return 0
    return max(lasts) - min(firsts)


def get_trace_broken_work_conservation_time(dat_file_path: str) -> float :
    print("Computing broken work conservation time for file :", dat_file_path)
    
    result = subprocess.run(
        ["/home/cgachod/repos/ocaml-scripts/running_waiting", "--wc", dat_file_path],
        stdout = subprocess.PIPE,
        universal_newlines = True
    )
    
    match = re.search(r'WC time.*?(\d+.\d+)', result.stdout)
    if match :
        return float(match.group(1))
    
def get_runs_dataframe(dir_path: str, use_cache: bool, cache_filename: str, sort_by_duration = True, include_broken_wc = True) -> pd.DataFrame :
    if not os.path.exists(dir_path) :
        return None

    # Policy is aggressive rewrite to cache
    cache_path = os.path.join(dir_path, cache_filename + ".csv")
    if use_cache and os.path.isfile(cache_path):
        df = pd.read_csv(cache_path)
        if sort_by_duration :
            df.sort_values('duration', ignore_index=True, inplace=True)
        if include_broken_wc and 'broken_wc_time' not in df.columns :
            df['broken_wc_time'] = df['filename'].map(lambda x: get_trace_broken_work_conservation_time(os.path.join(dir_path, x)))
            df.to_csv(cache_path, encoding='utf-8', index=False)
        df['exists'] = df['filename'].map(lambda x: os.path.isfile(os.path.join(dir_path, x)))
    else :
        dat_files = [file for file in os.listdir(dir_path) if file.endswith('.dat')]
        ndat = len(dat_files)
        if ndat == 0 :
            print(f"get_runs_dataframe : no trace files found in {dir_path}, skipping")
            return None
        
        df = pd.DataFrame({'filename': dat_files})
        df['duration'] = df['filename'].map(lambda x: get_trace_duration(os.path.join(dir_path, x)))
        if include_broken_wc :
            df['broken_wc_time'] = df['filename'].map(lambda x: get_trace_broken_work_conservation_time(os.path.join(dir_path, x)))
        df.sort_values('duration', ignore_index=True, inplace=True)
        df.to_csv(cache_path, encoding='utf-8', index=False)
        df['exists'] = True
    
    df['run_index'] = df['filename'].map(get_run_index)
    min_duration = df['duration'].min()
    df['variation'] = (df['duration'] - min_duration) * 100 / min_duration
    return df



def compute_cache_for_batch(dir_path: str, cache_filename: str, include_broken_wc = True) -> None :
    if not os.path.exists(dir_path) :
        return None

    # Policy is aggressive rewrite to cache
    cache_path = os.path.join(dir_path, cache_filename + ".csv")
    if os.path.isfile(cache_path):
        if not include_broken_wc:
            print(f"Skipping cache computation for batch in {dir_path} because it already exists and broken work conservation computation not requested")
            return
        df = pd.read_csv(cache_path)
    else:
        print(f"Computing runtime f{' and broken work conservation' if include_broken_wc else ''} cache for batch in {dir_path}")
        dat_files = [file for file in os.listdir(dir_path) if file.endswith('.dat')]
        ndat = len(dat_files)
        if ndat == 0 :
            print(f"get_runs_dataframe : no trace files found in {dir_path}, skipping")
            return None
        
        df = pd.DataFrame({'filename': dat_files})
        df['duration'] = df['filename'].map(lambda x: get_trace_duration(os.path.join(dir_path, x)))
    if include_broken_wc :
        df['broken_wc_time'] = df['filename'].map(lambda x: get_trace_broken_work_conservation_time(os.path.join(dir_path, x)))
    df.sort_values('duration', ignore_index=True, inplace=True)
    df.to_csv(cache_path, encoding='utf-8', index=False)
    

def compute_caches_for_benchmark(dir_path: str, cache_filename: str, include_broken_wc = True) -> None :
    print(f"Computing runtime {' and broken work conservation' if include_broken_wc else ''} cache for benchmark in {dir_path}")
    contents = os.listdir(dir_path)
    for c in contents :
        inner_dir_path = os.path.join(dir_path, c)
        if not os.path.isdir(inner_dir_path) :
            continue
        
        compute_cache_for_batch(
            inner_dir_path, 
            cache_filename, 
            include_broken_wc
        )
        
        
        
def compute_caches_for_repo(dir_path: str, cache_filename: str, include_broken_wc = True) -> None :
    print(f"Computing runtime {' and broken work conservation' if include_broken_wc else ''} cache for benchmark in {dir_path}")
    contents = os.listdir(dir_path)
    for c in contents :
        inner_dir_path = os.path.join(dir_path, c)
        if not os.path.isdir(inner_dir_path) :
            continue
        
        compute_caches_for_benchmark(
            inner_dir_path, 
            cache_filename, 
            include_broken_wc
        )








