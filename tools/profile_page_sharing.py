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
    

