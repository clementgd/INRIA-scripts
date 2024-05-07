import pandas as pd

def parse_with_latency(data_file_path: str, executable = "cg.C.x") -> pd.DataFrame:
    basic_info_regex_lat_str = r"^ *\S+ +(\d+) +(\d+) +(\d+\.\d+): +(\d+) +(\S+): +([0-9a-f]+)"
    data_src_regex_lat_str = r"[0-9a-f]+ \|OP (?:LOAD|STORE)\|([^\|]+)\|[^\|]+\|(TLB [^\|]+)\|[^\|]+\|[a-zA-Z\/\- ]+(\d+) +\d+ +([0-9a-f]+).+ ([0-9a-f]+)"

    command_str = f"perf script -i {data_file_path} -L -c {executable} --time 10%-20% > {PERF_SCRIPT_RESULTS_FILEPATH}"
    print(command_str)
    result = subprocess.run(
        command_str,
        shell=True,
        stdout = subprocess.PIPE,
        universal_newlines = True
    )