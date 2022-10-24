### Monitor class

import os
import logging
from threading import Thread
import psutil

try:
    import pynvml
except:
    logging.warning("pynvml does not load, no monitoring available")
import time

class Monitor(object):
    """ Class that monitors GPU utilization for a given time """

    def __init__(self, sampling_rate=None, pid=None):
        self.sampling_rate = sampling_rate if sampling_rate is not None else 0.1
        self.pid = pid if pid is not None else os.getpid()
        
        self.gpu = None
        self.should_stop = False
        self.thread = None
        self.accounting_enabled = False
        self.stats = []
    
    def _find_gpu(self):
        device_count = pynvml.nvmlDeviceGetCount()
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            for gpu_process in gpu_processes:
                if gpu_process.pid == self.pid:
                    self.gpu = handle
        #self.accounting_enabled = True
        self.accounting_enabled = pynvml.nvmlDeviceGetAccountingMode(self.gpu) == pynvml.NVML_FEATURE_ENABLED
        
        # Clear accounting statistics (requires root privileges)
        #pynvml.nvmlDeviceSetAccountingMode(self.gpu, pynvml.NVML_FEATURE_DISABLED)
        #pynvml.nvmlDeviceSetAccountingMode(self.gpu, pynvml.NVML_FEATURE_ENABLED)

    def _monitor(self):
        pynvml.nvmlInit()
        self._find_gpu()
        #self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        current_sample = []
        while not self.should_stop:
            used_cpu = None
            used_cpumem = None
            used_gpu = None
            used_gpumem = None
            
            cpu_process = psutil.Process(self.pid)
            used_cpu = cpu_process.cpu_percent(self.sampling_rate) / psutil.cpu_count() # CPU utilization in %
            used_cpumem = cpu_process.memory_info().rss // (1024*1024) # Memory use in MB

            gpu_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(self.gpu)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu)
            used_gpumem = memory.used  / 1024 / 1024
            #for gpu_process in gpu_processes:
            #    if gpu_process.pid == self.pid:
            #        used_gpumem = gpu_process.usedGpuMemory // (1024*1024) # GPU memory use in MB
            #        break

            if self.accounting_enabled:
                try:
                    stats = pynvml.nvmlDeviceGetAccountingStats(self.gpu, self.pid)
                    used_gpu = stats.gpuUtilization
                except pynvml.NVMLError: # NVMLError_NotFound
                    pass

            if not used_gpu:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu)
                used_gpu = util.gpu / len(gpu_processes) # Approximate based on number of processes

            current_sample.append((used_cpu, used_cpumem, used_gpu, used_gpumem))

            time.sleep(self.sampling_rate)

        #self.stats.append([round(sum(x) / len(x)) for x in zip(*current_sample)])
        #self.stats.append(current_sample)
        self.stats = [max(x) for x in zip(*current_sample)]
        pynvml.nvmlShutdown()
    
    def start_monitor(self):
        self.should_stop = False
        self.thread = Thread(target=self._monitor)
        self.thread.start()
    
    def stop_monitor(self):
        self.should_stop = True
        self.thread.join()
    
    def get_stats(self):
        return self.stats