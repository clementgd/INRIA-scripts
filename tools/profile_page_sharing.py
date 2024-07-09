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


def compute_page_sharing_stats(run_data: RunMemoryData):
    allocation_df = run_data.alloc_df
    allocation_df = allocation_df.loc[allocation_df["order"] > 0]
    n_filtered_allocations = len(allocation_df)
    print(f"Filtered out {len(run_data.alloc_df)} single page allocations ({n_filtered_allocations} remaining)")
    allocation_df = allocation_df.sort_values("pfn", ignore_index=True)
    print("Sorted allocations")

    access_df = run_data.access_df.sort_values("phys", ignore_index=True)
    print(f"Sorted {len(access_df) / 1e6} million accesses")
    access_df["pfn"] = access_df["phys"].values >> run_data.page_size_order
    print("Computed accesses pfns")
    
    access_idx = 0

    alloc_pfns = allocation_df["pfn"].to_numpy()
    alloc_orders = allocation_df["order"].to_numpy()
    access_pfns = access_df["pfn"].to_numpy()
    access_cpus = access_df["cpuid"].to_numpy()
    access_nodes = access_df["cpu_node"].to_numpy()

    previous_alloc_pfn = -1
    
    n_allocations_not_accessed = 0
    not_shared_allocations_pfns = []
    # For each allocation, how many small pages are shared
    n_node_shared_pages_per_allocation_pfn = {}
    n_cpu_shared_pages_per_allocation_pfn = {}

    for alloc_pfn, order in zip(alloc_pfns, alloc_orders):
        if alloc_pfn == previous_alloc_pfn:
            continue
        previous_alloc_pfn = alloc_pfn
        max_alloc_pfn = alloc_pfn + (1 << order)

        access_idx = bisect_left(access_pfns, alloc_pfn, lo = access_idx)
        if access_pfns[access_idx] >= max_alloc_pfn:
            n_allocations_not_accessed += 1
            continue
        
        # TODO Change it to cpu shared to see what it changes
        next_access_idx = bisect_left(access_pfns, max_alloc_pfn, lo = access_idx)
        unique_nodes = np.unique(access_nodes[access_idx:next_access_idx]) # TODO Make sure we are not including the next index
        if len(unique_nodes) <= 1:
            # Page connot be false shared if it is not shared
            not_shared_allocations_pfns.append(alloc_pfn)
            continue

        n_node_shared_pages = 0
        n_cpu_shared_pages = 0

        curr_pfn = -1
        curr_cpu = -1
        curr_node = -1
        is_page_cpu_shared = False
        is_page_node_shared = False
        while access_idx < next_access_idx:
            if access_pfns[access_idx] != curr_pfn:
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
            elif access_cpus[access_idx] != curr_cpu:
                is_page_cpu_shared = True
                
            access_idx += 1
            
        n_node_shared_pages_per_allocation_pfn[alloc_pfn] = n_node_shared_pages
        n_cpu_shared_pages_per_allocation_pfn[alloc_pfn] = n_cpu_shared_pages
    
    return not_shared_allocations_pfns, n_node_shared_pages_per_allocation_pfn, n_cpu_shared_pages_per_allocation_pfn, n_filtered_allocations
    


    
def collect_samples(program_path: str, results_dir_path: str, warmups: int = 1, capture_l3_miss_local_dram = True):
    # home_directory = "/root/tests"
    save_dir_path = get_benchmark_dir(results_dir_path, program_path)
    os.makedirs(save_dir_path, exist_ok=True)
    mem_events_period = 2000
    l3_miss_events_period = 200
    
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
        logging.info(f"Collecting samples into {data_file_name}")
        
        output_file_name = os.path.splitext(data_file_name)[0] + ".output.txt"
        measure_command = f"perf record -v -a -d -W -e {alloc_events} -e {mem_events} -c {l3_miss_events_period} -e {l3_miss_events} --output {data_file_name} --phys-data {program_path} &> {output_file_name}"
        command_chain = CommandChain([f"cd {save_dir_path}"])
        command_chain += init_command_chain_with_config(nb_on, omp_places)
        command_chain += CommandChain([program_path] * warmups)
        command_chain.append(measure_command)
        command_chain.execute()
        
    logging.info(f"Persisting machine info in {save_dir_path}")
    persist_machine_info(save_dir_path)
    run_benchmark_for_setup(False, get_sequential_omp_places(), "perf-mem-all-sequential.data")
    run_benchmark_for_setup(False, get_sockorder_omp_places(), "perf-mem-all-sockorder.data")
    


if __name__ == "__main__":
    # Set up basic configuration for logging
    logging.basicConfig(
        level=logging.INFO,
        format='\n### PY ### %(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'  # Only display hours, minutes, and seconds
    )
    
    RESULTS_DIR_PATH = "/tmp/perf_mem"
        
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", metavar="PROGRAM_PATH", help="The program to run and analyze")
    group.add_argument('--analyze', metavar="DATA_PATH", help='The data of a run to be analyzed')
    parser.add_argument('--mem_period', help='Number of runs to perform for each configuration', type=int, default=10)
    parser.add_argument('--miss_period', help='Number of warmups to perform for each configuration', type=int, default=1)
    args = parser.parse_args()
    
    collect_samples(args.run, RESULTS_DIR_PATH)
    
    # We also want to be able to analyze and run actually
    # collect_samples("/root/npb/NPB3.4-OMP/bin/cg.B.x")
    

