#!/root/jupyter/python_env/bin/python3

from typing import Optional, Tuple, Dict, List
from experiments_lib import (
    get_sequential_omp_places, get_sockorder_omp_places, 
    CommandChain, 
    init_command_chain_with_config,
    persist_machine_info, get_benchmark_dir
)
import logging
import argparse
import os
from time import process_time_ns
from bisect import bisect_left
from perf_lib import CustomPerfParser, RunMemoryData, filter_in_bounds, filter_cpu, bounds_for_object
import numpy as np
import pandas as pd


results_filename = "profiler_results.txt"


def compare_numbers(first: int, second: int):
    def show(number):
        if number < 1e3: 
            return f"{number}"
        else: 
            return f"{number:.3e}"
        
    return f"{show(first)} / {show(second)} : {(first * 100 / second):.1f}%"


def compute_page_sharing_stats(run_data: RunMemoryData):
    allocation_df = run_data.alloc_df
    allocation_df = allocation_df.loc[allocation_df["order"] > 0]
    total_filtered_allocations = len(allocation_df)
    print(f"Filtered out {len(run_data.alloc_df)} single page allocations ({total_filtered_allocations} remaining)")
    allocation_df = allocation_df.sort_values("pfn", ignore_index=True)
    print("Sorted allocations")

    access_df = run_data.access_df.sort_values("phys", ignore_index=True)
    print(f"Sorted {len(access_df) / 1e6} million accesses")
    access_df["pfn"] = access_df["phys"].values >> run_data.page_size_order
    print("Computed accesses pfns")
    access_df["is_l3_remote_dram"] = access_df["event"] == "mem_load_l3_miss_retired.remote_dram:P"
    total_l3_remote_dram = np.count_nonzero(access_df["is_l3_remote_dram"])
    
    access_idx = 0

    alloc_pfns = allocation_df["pfn"].to_numpy()
    alloc_orders = allocation_df["order"].to_numpy()
    access_pfns = access_df["pfn"].to_numpy()
    access_cpus = access_df["cpuid"].to_numpy()
    access_nodes = access_df["cpu_node"].to_numpy()
    access_are_l3_remote_dram = access_df["is_l3_remote_dram"].to_numpy()

    previous_alloc_pfn = -1
    
    pfn_column = []
    order_column = []
    n_unique_nodes_column = []
    n_unique_cpus_column = []
    n_node_shared_pages_column = []
    n_cpu_shared_pages_column = []
    n_l3_remote_dram_column = []
    
    for alloc_pfn, order in zip(alloc_pfns, alloc_orders):
        if alloc_pfn == previous_alloc_pfn:
            continue
        previous_alloc_pfn = alloc_pfn
        max_alloc_pfn = alloc_pfn + (1 << order)

        access_idx = bisect_left(access_pfns, alloc_pfn, lo = access_idx)
        if access_pfns[access_idx] >= max_alloc_pfn:
            # No memory accesses for this allocation
            continue

        # print(f"\nAllocation {i}, n pages = {1 << order}, alloc_pfn = {hex(alloc_pfn)}, max_alloc_pfn = {hex(max_alloc_pfn)}")
        # print(f"Skipped {access_idx - initial_access_idx} accesses")
        
        pfn_column.append(alloc_pfn)
        order_column.append(order)

        # TODO If order is 1 then page is truly shared as soon as several nodes are accessing
        next_access_idx = bisect_left(access_pfns, max_alloc_pfn, lo = access_idx)
        n_l3_remote_dram = np.count_nonzero(access_are_l3_remote_dram[access_idx:next_access_idx])
        # print(f"n_l3_remote_dram : {n_l3_remote_dram}")
        n_l3_remote_dram_column.append(n_l3_remote_dram)
        
        unique_nodes = np.unique(access_nodes[access_idx:next_access_idx])
        n_unique_nodes_column.append(len(unique_nodes))
        unique_cpus = np.unique(access_cpus[access_idx:next_access_idx])
        n_unique_cpus_column.append(len(unique_cpus))
        if len(unique_cpus) <= 1:
            n_node_shared_pages_column.append(0)
            n_cpu_shared_pages_column.append(0)
            # Page connot be false shared if it is not shared
            continue

        # print(f"Access pfn : {hex(access_pfns[access_idx])}, next access pfn : {hex(access_pfns[next_access_idx])}")

        n_node_shared_pages = 0
        n_cpu_shared_pages = 0

        curr_pfn = -1
        curr_cpu = -1
        curr_node = -1
        is_page_cpu_shared = False
        is_page_node_shared = False
        # We want to count the number of pages that are shared either just by CPUs or between nodes
        while access_idx < next_access_idx:
            if access_pfns[access_idx] != curr_pfn:
                # print(f"New pfn : {access_pfns[access_idx]}")
                if is_page_node_shared:
                    n_cpu_shared_pages += 1
                    n_node_shared_pages += 1
                elif is_page_cpu_shared:
                    n_cpu_shared_pages += 1
                
                curr_pfn = access_pfns[access_idx]
                curr_cpu = access_cpus[access_idx]
                curr_node = access_nodes[access_idx]
                is_page_cpu_shared = False
                is_page_node_shared = False
                
            elif access_nodes[access_idx] != curr_node:
                is_page_node_shared = True
                # Nothing to see on that page anymore
                access_idx = bisect_left(access_pfns, curr_pfn + 1, lo = access_idx)
                # print(f"Page is node shared. Curr pfn : {curr_pfn}, new pfn : {access_pfns[access_idx]}")
                continue
                
            elif access_cpus[access_idx] != curr_cpu:
                is_page_cpu_shared = True
                # print(f"Page is cpu shared : {curr_cpu} and {access_cpus[access_idx]}")
                
            access_idx += 1
            
        n_node_shared_pages_column.append(n_node_shared_pages)
        n_cpu_shared_pages_column.append(n_cpu_shared_pages)
            
    stats_df = pd.DataFrame({
        'pfn': pfn_column,
        'order': order_column,
        'n_unique_nodes': n_unique_nodes_column,
        'n_unique_cpus': n_unique_cpus_column,
        'n_node_shared_pages': n_node_shared_pages_column,
        'n_cpu_shared_pages': n_cpu_shared_pages_column,
        'n_l3_remote_dram': n_l3_remote_dram_column
    })
        
    return stats_df, total_l3_remote_dram
    


    
def collect_samples(program_path: str, results_dir_path: str, warmups: int = 1, 
                    capture_l3_miss_local_dram = True, mem_events_period = 2000, 
                    l3_miss_events_period = 200):
    # home_directory = "/root/tests"
    executable = os.path.basename(program_path)
    save_dir_path = get_benchmark_dir(results_dir_path, program_path)
    os.makedirs(save_dir_path, exist_ok=True)
    
    mem_events = f"cpu/mem-loads,period={mem_events_period}/P,cpu/mem-stores,period={mem_events_period}/P"
    
    # Following events will always be recorded with a period of 1
    alloc_events = "kmem:\*,syscalls:sys_enter_mmap"
    
    l3_miss_events = (
        "mem_load_l3_miss_retired.remote_dram:P,"
        "mem_load_l3_miss_retired.remote_fwd:P,"
        "mem_load_l3_miss_retired.remote_hitm:P"
    )
    
    if capture_l3_miss_local_dram:
        l3_miss_events += ",mem_load_l3_miss_retired.local_dram:P"
    
    def run_benchmark_for_setup(nb_on: bool, omp_places: Optional[str], data_file_name: str):
        logging.info(f"Collecting samples into file {data_file_name}")
        
        output_file_name = os.path.splitext(data_file_name)[0] + ".output.txt"
        # measure_command = f"perf record -v -a -d -W -e {alloc_events} -e {mem_events} -c {l3_miss_events_period} -e {l3_miss_events} --output {data_file_name} --phys-data {program_path} &> {output_file_name}"
        measure_command = f"perf record -v -a -d -W -e {alloc_events} -e {mem_events} -c {l3_miss_events_period} -e {l3_miss_events} --output {data_file_name} --phys-data {program_path} 2>&1 | tee {output_file_name}"
        
        command_chain = CommandChain([f"cd {save_dir_path}"])
        command_chain += init_command_chain_with_config(nb_on, omp_places)
        command_chain.append(f"echo Running {warmups} warmups")
        command_chain += CommandChain([program_path] * warmups)
        command_chain.append(rf'printf "\n==========================\nNow running actual measurements\nOutput saved into $(pwd)/{output_file_name}\n\n"')
        command_chain.append(measure_command)
        # command_chain.append(f"cat $(pwd)/{output_file_name}")
        command_chain.append(rf'printf "\nResults saved into $(pwd)/{data_file_name}"')
        command_chain.execute()
        
    logging.info(f"Persisting machine info in {save_dir_path}")
    persist_machine_info(save_dir_path)
    run_benchmark_for_setup(False, get_sequential_omp_places(), f"perf-sequential-{executable}.data")
    run_benchmark_for_setup(False, get_sockorder_omp_places(), f"perf-sockorder-{executable}.data")
    run_benchmark_for_setup(False, None, f"perf-none-{executable}.data")
    
    return save_dir_path







def print_stats(stats_df: pd.DataFrame, total_l3_remote_dram: int, file_name: str):
    all_node_shared_df = stats_df.loc[stats_df["n_unique_nodes"] > 1]
    node_false_shared_df = all_node_shared_df.loc[all_node_shared_df["n_node_shared_pages"] == 0]
    n_all_node_shared_allocs = len(all_node_shared_df)
    n_false_node_shared_allocs = len(node_false_shared_df)
    
    all_cpu_shared_df = stats_df.loc[stats_df["n_unique_cpus"] > 1]
    all_cpu_false_shared_df = all_cpu_shared_df.loc[all_cpu_shared_df["n_cpu_shared_pages"] == 0]
    n_all_cpu_shared_allocs = len(all_cpu_shared_df)
    n_false_cpu_shared_allocs = len(all_cpu_false_shared_df)
    
    output = ""
    output += f"\n\n###### Statistics for data file : {file_name} ######\n"
    output += f"Number of NODE false shared / all NODE shared (true + false) large allocations : {compare_numbers(n_false_node_shared_allocs, n_all_node_shared_allocs)}\n"
    output += f"Number of CPU false shared / all CPU shared (true + false) large allocations : {compare_numbers(n_false_cpu_shared_allocs, n_all_cpu_shared_allocs)}\n"
    
    # l3_remote_dram_counts = np.array(list(stats.n_l3_remote_dram_per_allocation_pfn.values()), dtype=int)
    
    # total_l3_remote_dram_false_shared = sum([stats.n_l3_remote_dram_per_allocation_pfn[pfn] for pfn, count in stats.n_node_shared_pages_per_allocation_pfn.items() if count == 0])
    total_l3_remote_dram_large_allocations = sum(stats_df["n_l3_remote_dram"])
    
    total_l3_remote_dram_all_node_shared = sum(all_node_shared_df["n_l3_remote_dram"])
    total_l3_remote_dram_false_node_shared = sum(node_false_shared_df["n_l3_remote_dram"])
    
    total_l3_remote_dram_all_cpu_shared = sum(all_cpu_shared_df["n_l3_remote_dram"])
    total_l3_remote_dram_false_cpu_shared = sum(all_cpu_false_shared_df["n_l3_remote_dram"])

    # total_l3_remote_dram_false_shared = sum(l3_remote_dram_counts[node_shared_counts == 0])
    # total_l3_remote_dram_true_shared = sum(l3_remote_dram_counts[node_shared_counts > 0])
    
    output += "\nNumber of mem_load_l3_miss_retired.remote_dram in :\n"
    output += f" - false NODE shared pages / all NODE shared pages -- in large allocations : {compare_numbers(total_l3_remote_dram_false_node_shared, total_l3_remote_dram_all_node_shared)}\n"
    output += f" - all NODE shared pages / all pages -- in large allocations : {compare_numbers(total_l3_remote_dram_all_node_shared, total_l3_remote_dram_large_allocations)}\n"
    output += f" - false CPU shared pages / all CPU shared pages  -- in large allocations : {compare_numbers(total_l3_remote_dram_false_cpu_shared, total_l3_remote_dram_all_cpu_shared)}\n"
    output += f" - all CPU shared pages / all pages -- in large allocations : {compare_numbers(total_l3_remote_dram_all_cpu_shared, total_l3_remote_dram_large_allocations)}\n"
    output += f" - large allocations / all allocations : {compare_numbers(total_l3_remote_dram_large_allocations, total_l3_remote_dram)}\n"
    output += "###### - ######\n\n"
    
    with open(results_filename, 'a') as f:
        f.write(output)
    print(output)




def analyze_file(file_path: str, perf_parser: Optional[CustomPerfParser] = None):
    if perf_parser is None:
        dir_path = os.path.dirname(file_path)
        perf_parser = CustomPerfParser(dir_path)
    
    file_name = os.path.basename(file_path)
    name, extension = os.path.splitext(file_name)
    if extension != ".data":
        return
    
    executable = name.split("-")[-1]
    run_data = perf_parser.extract_and_read(
        file_path, force_rerun_extraction=False, executable=executable)
    stats_df, total_l3_remote_dram = compute_page_sharing_stats(run_data)
    print_stats(stats_df, total_l3_remote_dram, file_name)
    

def analyze_dir(dir_path: str):
    perf_parser = CustomPerfParser(dir_path)
    contents = os.listdir(dir_path)
    for file_name in contents:
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path):
            analyze_file(file_path, perf_parser)
        




if __name__ == "__main__":
    # Set up basic configuration for logging
    logging.basicConfig(
        level=logging.INFO,
        format='\n### PY ### %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'  # Only display hours, minutes, and seconds
    )
    
    RESULTS_DIR_PATH = "/tmp/perf"
        
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", metavar="PROGRAM_PATH", help="The program to run and analyze")
    group.add_argument('--analyze_file', metavar="DATA_PATH", help='The data of a run to be analyzed')
    group.add_argument('--analyze_dir', metavar="DATA_PATH", help='The data of a run to be analyzed')
    parser.add_argument('--mem_period', help='Sampling period for generic memory events', type=int, default=2000)
    parser.add_argument('--miss_period', help='Sampling period for mem_load_l3_miss_retired events', type=int, default=200)
    parser.add_argument('--nwarmups', help='Number of warmups to perform for each configuration', type=int, default=1)
    args = parser.parse_args()
    
    if args.analyze_file is not None:
        analyze_file(args.analyze_file)
        exit()
        
    if args.run is not None:
        save_dir_path = collect_samples(args.run, RESULTS_DIR_PATH, args.nwarmups, False, args.mem_period, args.miss_period)
    else:
        save_dir_path = args.analyze_dir
        
    analyze_dir(save_dir_path)

    
# ./profile_page_sharing.py --analyze_dir /tmp/perf/cg.C.x__dahu-14__v6.8.0-rc3__performance__2024-07-11

# ./profile_page_sharing.py --nwarmups 2  --run /root/npb/NPB3.4-OMP/bin/cg.C.x


# ./profile_page_sharing.py --analyze_dir /tmp/perf/cg.C.x__dahu-32__v6.8.0-rc3__performance__2024-07-10



    

