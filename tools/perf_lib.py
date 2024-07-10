import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Union, Callable
import pandas as pd
import os
import re
import subprocess
from collections import defaultdict, namedtuple
import math as ma
from dataclasses import dataclass



def filter_in_bounds(df: pd.DataFrame, bounds: tuple, variable: str) -> pd.DataFrame:
    if bounds[1] is None:
        return df.loc[df[variable] >= bounds[0]]
    if bounds[0] is None:
        return df.loc[df[variable] < bounds[1]]
    return df.loc[(df[variable] >= bounds[0]) & (df[variable] < bounds[1])]

def filter_cpu(df: pd.DataFrame, cpuid: Union[int, List[int]]) -> pd.DataFrame:
    if isinstance(cpuid, list):
        filtered_df = df[df['cpuid'].isin(cpuid)]
    else:
        filtered_df = df[df['cpuid'] == cpuid]
    return filtered_df

def filter_local_access(df: pd.DataFrame):
    return df.loc[df['cpu_node'] == df['memory_node']]

def filter_remote_access(df: pd.DataFrame):
    return df.loc[df['cpu_node'] != df['memory_node']]

def filter_loads_stores(df: pd.DataFrame):
    return df.loc[df['event'].str.contains("cpu/")]

def local_access_proportion(df: pd.DataFrame):
    # (sock_df['cpu_node'] == sock_df['memory_node']).astype(int) * sock_df['period']
    return 100 * sum((df['cpu_node'] == df['memory_node']).astype(int) * df['period']) / sum(df['period'])



def find_virt_page_for_pfn(pfn: int, order: int, access_df: pd.DataFrame, page_size: int, page_size_order: int):
    min_phys_addr = pfn * page_size
    max_phys_addr = (pfn + 2 ** order) * page_size
    
    access_to_zone_df = filter_in_bounds(access_df, (min_phys_addr, max_phys_addr), "phys")
    # print(len(access_to_zone_df))
    if len(access_to_zone_df) == 0:
        return None
    access_record = access_to_zone_df.loc[access_to_zone_df['virt'].idxmin()]
    phys_pfn = int(access_record['phys']) >> page_size_order
    virt_page = int(access_record['virt']) >> page_size_order
    return virt_page - (phys_pfn - pfn)





def get_shell_command_output(command: str):
    return subprocess.run(
        command,
        shell=True, stdout = subprocess.PIPE, universal_newlines = True
    ).stdout.strip('\n')


# TODO Move the machine data like page size into here
@dataclass
class RunMemoryData:
    nodes_phys_mem_upper_boundaries: List[int]
    access_df: pd.DataFrame
    alloc_df: pd.DataFrame
    mmap_df: pd.DataFrame
    # objects_addr_df: pd.DataFrame
    page_size: int # The standard page size
    page_size_order: Optional[int]
    
# TODO : There should be a file alongside the perf data file that contains all that info
# TODO : Why need for a class ?
# -> This class because lots of functions needed to parse and allows to group them
class CustomPerfParser:
    
    def __init__(self, home_dir: str) -> None:
        self.home_dir = home_dir
        self.process_metadata(self.home_dir)
    
    
    def process_metadata(self, dir_path: Optional[str] = None):
        if dir_path is None:
            dir_path = self.home_dir
        metadata_dir_path = os.path.join(dir_path, "meta")
        with open(f"{metadata_dir_path}/lscpu.log", 'r') as f:
            lscpu_log = f.read()
        with open(f"{metadata_dir_path}/pagesize.log", 'r') as f:
            page_size_log = f.read()
        with open(f"{metadata_dir_path}/zoneinfo.log", 'r') as f:
            zoneinfo_log = f.read()
            
        self.page_size = int(page_size_log)
        self.page_size_order = int(ma.log2(self.page_size))
            
            
        self.cpuid_to_node = {}
        matches = re.findall(
            r"NUMA +node\d+ +CPU\(s\): +([\d,]+)",
            lscpu_log
        )

        self.nodes_cpuids = []

        node_id = 0
        for cpulist in matches:
            integer_cpulist = [int(cpuid) for cpuid in cpulist.split(',')]
            self.nodes_cpuids.append(integer_cpulist)
            for cpuid in integer_cpulist:
                self.cpuid_to_node[cpuid] = node_id
            node_id += 1
            
        
        matches = re.findall(
            r"start_pfn: +(\d+)\sNode (\d+), +zone +(\w+)", 
            zoneinfo_log
        )

        self.node_upper_boundaries = []
        for m in matches:
            if int(m[1]) > len(self.node_upper_boundaries):
                self.node_upper_boundaries.append(int(m[0]) * self.page_size)
                
        print(f"Node upper boundaries : {self.node_upper_boundaries}")
        
        
    def get_node_for_physical_address(self, addr: int):
        boundary_idx = 0
        while boundary_idx < len(self.node_upper_boundaries) and addr >= self.node_upper_boundaries[boundary_idx]:
            boundary_idx += 1
        return boundary_idx 


    # Calls perf's extraction command from the shell
    # Produces an extracted file and returns the path to that file
    def extract_perf_data_file(self, file_name: str, dir_path: Optional[str] = None, 
                               force_rerun_extraction: Optional[bool] = False, 
                               executable: Optional[str] = None, 
                               time_option: Optional[str] = None):
        if dir_path is None:
            dir_path = self.home_dir
        file_path = os.path.join(dir_path, file_name)
        
        sha1sum = get_shell_command_output(f"sha1sum {file_path}")
        # print(f"Computed checksum : {sha1sum}")
        extracted_file_path = file_path + ".log"
                
        if not force_rerun_extraction and os.path.isfile(extracted_file_path):
            with open(extracted_file_path, 'r') as f:
                first_line = f.readline().strip("\n")
            if first_line == sha1sum:
                print("Matching checksum found, skipping extraction")
                return extracted_file_path
        
        os.makedirs(os.path.dirname(extracted_file_path), exist_ok=True)
        with open(extracted_file_path, 'w') as f:
            f.write(f"{sha1sum}\n")
            
        exec_filter = f"-c {executable}" if executable is not None else ""
        time_filter = f"--time {time_option}" if time_option is not None else ""
        command_str = (f"perf script -i {file_path} -L -F +period "
                       f"{exec_filter} {time_filter} >> {extracted_file_path}")
        print(f"Executing extraction command : {command_str}")
        result = subprocess.run(
            command_str,
            shell=True,
            stdout = subprocess.PIPE,
            universal_newlines = True
        )
        # print(f"Output : {result.stdout}")
        
        return extracted_file_path
        
    
    
    def read_kmem_events(self, extracted_file_path: str, 
                         get_node_info = False) -> pd.DataFrame:
        # 1: pid, 2: cpu, 3: timestamp, 4: period, 5: event
        kmem_regex = re.compile(r"^ *\S+ +(\d+) +(\d+) +(\d+\.\d+): +(\d+) +(kmem:\S*):")
        
        cpuids = []
        times = []
        periods = []
        events = []
        
        with open(extracted_file_path) as f :
            for line in f :
                matched = kmem_regex.match(line)
                if matched:
                    cpuids.append(int(matched[2]))
                    times.append(float(matched[3]))
                    periods.append(int(matched[4]))
                    events.append(matched[5])
                    continue
                
        kmem_df = pd.DataFrame({
            "cpuid": cpuids,  
            "time": times, 
            "period": periods,
            "event": events,
        })
        
        if get_node_info:
            kmem_df['cpu_node'] = kmem_df['cpuid'].map(self.cpuid_to_node)
            kmem_df['memory_node'] = kmem_df['phys'].map(self.get_node_for_physical_address)
        return kmem_df



    def get_event_counts(extracted_file_path: str):
        # 1: pid, 2: cpu, 3: timestamp, 4: period, 5: event
        kmem_regex = re.compile(r"^ *\S+ +(\d+) +(\d+) +(\d+\.\d+): +(\d+) +(\S+):")
        event_counts = defaultdict(int)
        with open(extracted_file_path) as f :
            for line in f :
                matched = kmem_regex.match(line)
                if matched:
                    event_counts[matched[5]] += 1
                    continue
        return event_counts
    
    
    
    def read_output_virtual_memory_ranges(self, file_path: Optional[str]):
        if file_path is None or not os.path.exists(file_path):
            return pd.DataFrame()
        
        virt_addr_regex = re.compile(r" +VIRT_ADDR of (\S+): +(\d+) +size: +(\d+).*")
        
        object_names = []
        virtual_addresses = []
        sizes = []
        
        with open(file_path, "r") as f :
            for line in f:
                matched = virt_addr_regex.search(line)
                if matched:
                    print(f"Matched line : {line}")
                    object_names.append(matched[1])
                    virtual_addresses.append(int(matched[2]))
                    sizes.append(int(matched[3]))
                    
        return pd.DataFrame({
            'name': object_names,
            'virt': virtual_addresses,
            'size': sizes
        })

        
    def read_mem_access_and_alloc_events(self, extracted_file_path: str, 
                                         output_file_path: str) -> RunMemoryData:
        # 1: pid, 2: cpu, 3: timestamp, 4: period, 5: event, 6: page, 7: pfn, 8: order
        page_alloc_regex = re.compile(
            r"^ *\S+ +(\d+) +(\d+) +(\d+\.\d+): +(\d+) +(kmem:mm_page_alloc\S*): "
            r"+page=0x([0-9a-f]+) +pfn=0x([0-9a-f]+) +order=(\d+)")

        # 1: pid, 2: cpuid, 3: timestamp, 4: period, 5: event, 6: virt_addr 
        basic_info_pattern = r"^ *\S+ +(\d+) +(\d+) +(\d+\.\d+): +(\d+) +(\S+): +([0-9a-f]+)"
        
        # 1: cache_result, 2: tlb_result, 3: latency, 4: phys_adress
        data_src_with_latency_pattern = (
            r"[0-9a-f]+ \|OP (?:LOAD|STORE)\|([^\|]+)\|[^\|]+\|(TLB [^\|]+)\|"
            r"[^\|]+\|[a-zA-Z\/\- ]+(\d+) +\d+ +[0-9a-f]+.+ ([0-9a-f]+)")
        access_with_latency_regex = re.compile(
            basic_info_pattern + r" +" + data_src_with_latency_pattern)
        
        # 1: pid, 2: cpu, 3: time, 4: addr, 5: len, 6: prot, 7: flags, 8: fd, 9: offset 
        enter_mmap_regex = re.compile(
            r"^ *\S+ +(\d+) +(\d+) +(\d+\.\d+): +\d+ +syscalls:sys_enter_mmap:"
            r" +addr: +0x([0-9a-f]+), +len: +0x([0-9a-f]+), +prot: +0x([0-9a-f]+), "
            r"flags: +0x([0-9a-f]+), +fd: +0x([0-9a-f]+), +off: +0x([0-9a-f]+)")
        
        access_cpuids = []
        access_timestamps = []
        access_periods = []
        access_events = []
        access_virtual_addrs = []
        access_cache_results = []
        access_latencies = []
        access_physical_addrs = []
        
        alloc_cpuids = []
        alloc_times = []
        alloc_events = []
        alloc_pfns = []
        alloc_orders = []
        
        mmap_cpuids = []
        mmap_times = []
        mmap_virtual_addrs = []
        mmap_lens = []
        mmap_prots = []
        mmap_flags = []
        mmap_fds = []
        mmap_offsets = []
        
        lines_count = 0
        not_matched_lines_count = 0
                
        with open(extracted_file_path) as f :
            for line in f :
                lines_count += 1
                matched = access_with_latency_regex.match(line)
                if matched:
                    phys_addr = int(matched[10], base=16)
                    if phys_addr == 0:
                        continue
                    access_cpuids.append(int(matched[2]))
                    access_timestamps.append(float(matched[3]))
                    access_periods.append(int(matched[4]))
                    access_events.append(matched[5])
                    access_virtual_addrs.append(int(matched[6], base=16))
                    access_cache_results.append(matched[7])
                    access_latencies.append(int(matched[9]))
                    access_physical_addrs.append(phys_addr)
                    continue
                
                matched = page_alloc_regex.match(line)
                if matched:
                    alloc_cpuids.append(int(matched[2]))
                    alloc_times.append(float(matched[3]))
                    alloc_events.append(matched[5])
                    alloc_pfns.append(int(matched[7], base=16))
                    alloc_orders.append(int(matched[8]))
                    continue
                
                matched = enter_mmap_regex.match(line)
                if matched:
                    mmap_cpuids.append(int(matched[2]))
                    mmap_times.append(float(matched[3]))
                    mmap_virtual_addrs.append(int(matched[4], base=16))
                    mmap_lens.append(int(matched[5], base=16))
                    mmap_prots.append(int(matched[6], base=16))
                    mmap_flags.append(int(matched[7], base=16))
                    mmap_fds.append(int(matched[8], base=16))
                    mmap_offsets.append(int(matched[9], base=16))
                
                if "kmem:" in line or "syscalls:" in line:
                    continue
                
                not_matched_lines_count += 1
        
        print(f"{not_matched_lines_count} lines not matched / {lines_count}")
        
        access_df = pd.DataFrame({
            "cpuid": access_cpuids,  
            "time": access_timestamps, 
            "period": access_periods,
            "event": access_events,
            "virt": access_virtual_addrs,
            "cache_result": access_cache_results,
            "latency": access_latencies,
            "phys": access_physical_addrs, 
        })
        
        alloc_df = pd.DataFrame({
            'cpuid': alloc_cpuids,
            'time': alloc_times,
            'event': alloc_events,
            'pfn': alloc_pfns,
            'order': alloc_orders
        })
        
        mmap_df = pd.DataFrame({
            'cpuid': mmap_cpuids,
            'time': mmap_times,
            'virt': mmap_virtual_addrs,
            'len': mmap_lens,
            'prot': mmap_prots,
            'flags': mmap_flags,
            'fd': mmap_fds,
            'offset': mmap_offsets
        })
        
        # More practical to do that than to try and extract initial time out 
        # of the first line that has a timestamp
        initial_timestamp = min(access_df['time'].min(), alloc_df['time'].min(), mmap_df['time'].min())
        access_df['time'] = access_df['time'] - initial_timestamp
        alloc_df['time'] = alloc_df['time'] - initial_timestamp
        mmap_df['time'] = mmap_df['time'] - initial_timestamp
        
        print("Computing physical node information")
        access_df['cpu_node'] = access_df['cpuid'].map(self.cpuid_to_node)
        access_df['memory_node'] = access_df['phys'].map(self.get_node_for_physical_address)
        alloc_df['cpu_node'] = alloc_df['cpuid'].map(self.cpuid_to_node)
        alloc_df['memory_node'] = alloc_df['pfn'].map(lambda x: self.get_node_for_physical_address(x * self.page_size))
        
        # print("Determining virtual pages corresponding with physical allocations (this may take several minutes)")
        # alloc_df['virt_page'] = alloc_df.apply(lambda x: find_virt_page_for_pfn(x.pfn, x.order, access_df, self.page_size, self.page_size_order), axis=1)
        
        # I guess what I could just do here is convert the allocation to their base address ?
        # TODO Get rid of objects memory ranges, plus they need to use a different version of the NAS bench
        
        # return RunMemoryData(self.node_upper_boundaries, access_df, alloc_df, mmap_df, self.read_output_virtual_memory_ranges(output_file_path))
        return RunMemoryData(self.node_upper_boundaries, access_df, alloc_df, mmap_df, self.page_size, self.page_size_order)
    
    
    def extract_and_read(self, file_name: str, dir_path: Optional[str] = None, map_alloc_virtual_pages = False, force_rerun_extraction = False, executable = "cg.C.x", time_option = None):
        print("Extracting perf data file. This can take a couple minutes...")
        extracted_file_path = self.extract_perf_data_file(file_name, dir_path, force_rerun_extraction, executable, time_option)
        if dir_path is None :
            dir_path = self.home_dir
        output_file_path = os.path.splitext(os.path.join(dir_path, file_name))[0] + ".output.txt"
        print("Parsing extracted perf data file. This can take a couple minutes...")
        return self.read_mem_access_and_alloc_events(extracted_file_path, output_file_path)
    
    
    
    

def bounds_for_object(config_data: RunMemoryData, object_name: str):
    res_df = config_data.objects_addr_df.loc[config_data.objects_addr_df['name'] == object_name]
    if len(res_df) == 0:
        return None
    else:
        start_addr = res_df.iloc[0]['virt']
        size_bytes = res_df.iloc[0]['size']
        return (start_addr, start_addr + size_bytes)