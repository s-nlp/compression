### Monitor class

import os
import logging
from threading import Thread
import psutil

from transformers.utils import is_py3nvml_available, is_torch_available, logging


try:
    import py3nvml.py3nvml as pynvml
except:
    logging.warning("pynvml does not load, no monitoring available")
import time

class Monitor(object):
    """ Class that monitors GPU utilization for a given time """

    def __init__(self, sampling_rate=None, pid=None):
        self.sampling_rate = sampling_rate if sampling_rate is not None else 0.01
        self.pid = pid if pid is not None else os.getpid()
        
        self.gpu = None
        self.should_stop = False
        self.thread = None
        self.accounting_enabled = False
        self.stats = []
    
    def _find_gpu(self):
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 1:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu = handle
        else:
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for gpu_process in gpu_processes:
                    if gpu_process.pid == self.pid:
                        self.gpu = handle

        self.accounting_enabled = pynvml.nvmlDeviceGetAccountingMode(self.gpu) == pynvml.NVML_FEATURE_ENABLED
        
        # Clear accounting statistics (requires root privileges)
        #pynvml.nvmlDeviceSetAccountingMode(self.gpu, pynvml.NVML_FEATURE_DISABLED)
        #pynvml.nvmlDeviceSetAccountingMode(self.gpu, pynvml.NVML_FEATURE_ENABLED)

    def _monitor(self):
        pynvml.nvmlInit()
        self._find_gpu()
        current_sample = []
        while not self.should_stop:
            used_cpu = 0
            used_cpumem = 0
            used_gpu = 0
            used_gpumem = 0

            cpu_process = psutil.Process(self.pid)

            used_cpu = cpu_process.cpu_percent(self.sampling_rate) / psutil.cpu_count() # CPU utilization in %
            used_cpumem = cpu_process.memory_info().rss // (1024*1024) # Memory use in MB

            #for proc in pynvml.nvmlDeviceGetComputeRunningProcesses(self.gpu):
            #    print(
            #        "pid %d using %d bytes of memory on device %d."
            #        % (proc.pid, proc.usedGpuMemory, self.pid)
            #    )


            gpu_processes = pynvml.nvmlDeviceGetComputeRunningProcesses(self.gpu)
            memory = pynvml.nvmlDeviceGetMemoryInfo(self.gpu)
            #used_gpumem = memory.used  / 1024 / 1024


            if self.accounting_enabled:
                try:
                    stats = pynvml.nvmlDeviceGetAccountingStats(self.gpu, self.pid)
                    used_gpu = stats.gpuUtilization
                except pynvml.NVMLError: # NVMLError_NotFound
                    pass

            if not used_gpu:
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu)
                try:
                    used_gpu = util.gpu / len(gpu_processes) # Approximate based on number of processes
                    for proc in gpu_processes:
                        if proc.pid == self.pid:
                            used_gpumem = proc.usedGpuMemory / 1024 / 1024
                except:
                    #nvmlDeviceGetUtilization is wrong
                    used_gpu, used_gpumem = -1, -1

            current_sample.append((used_cpu, used_cpumem, used_gpu, used_gpumem))

            time.sleep(self.sampling_rate)

        #self.stats.append([round(sum(x) / len(x)) for x in zip(*current_sample)])
        #self.stats.extend(current_sample)
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
        #return [max(x) for x in zip(*self.stats)]
        return self.stats