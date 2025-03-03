import pynvml
import time

gpu_id = 0
pynvml.nvmlInit()
used_max = 0
while True:
    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    used = round(meminfo.used / 1024 / 1024, 2)
    if used > used_max:
        used_max = used
        print(used_max)
    time.sleep(1)
print(used_max)
