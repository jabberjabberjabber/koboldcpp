import os
import re
import sys
import platform
import subprocess

from . import state
from . import utils


def get_default_threads():
    physical_core_limit = 1
    if os.cpu_count() is not None and os.cpu_count()>1:
        physical_core_limit = os.cpu_count() // 2
    default_threads = (physical_core_limit if physical_core_limit<=3 else max(3,physical_core_limit-1))
    processor = platform.processor()
    if 'Intel' in processor:
        default_threads = (8 if default_threads > 8 else default_threads) #this helps avoid e-cores.
    if default_threads > 64:
        print(f"Auto CPU Threads capped at 64 (instead of {default_threads}). You can override this by passing an explicit number of --threads.")
        default_threads = 64
    return default_threads


def autoset_gpu_layers(ctxsize, sdquanted, bbs, musiclowvram): #shitty algo to determine how many layers to use
    gpumem = state.MaxMemory[0]
    usedmem = 0
    if state.MaxFreeMemory[0]>0:
        usedmem = state.MaxMemory[0]-state.MaxFreeMemory[0]
        if state.showusedmemwarning and usedmem > (2.5*1024*1024*1024):
            state.showusedmemwarning = False
            print(f"Note: KoboldCpp has detected that a significant amount of GPU VRAM ({usedmem/1024/1024} MB) is currently used by another application.\nFor best results, you may wish to close that application and then restart KoboldCpp.\n***")
    reservedmem = max(1.25*1024*1024*1024,(0.5*1024*1024*1024 + usedmem)) # determine vram overhead
    try:
        if not state.modelfile_extracted_meta:
            return 0
        layerlimit = 0
        fsize = state.modelfile_extracted_meta[2]
        fname = state.modelfile_extracted_meta[0]
        if fsize > (10*1024*1024): #dont bother with models < 10mb
            cs = ctxsize
            mem = gpumem
            if "-00001-of-00" in fname:
                match = re.search(r'-(\d{5})-of-(\d{5})\.', fname)
                if match:
                    total_parts = int(match.group(2))
                    if total_parts > 1 and total_parts <= 999:
                        if state.showmultigpuwarning:
                            state.showmultigpuwarning = False
                            print("Multi-Part GGUF detected. Layer estimates may not be very accurate - recommend setting layers manually.")
                        fsize *= total_parts

            state.calulated_gpu_overhead = 0
            musicoh1 = 0
            musicoh2 = 0
            if state.modelfile_extracted_meta[3] > 1024*1024*1024*5: #sdxl tax
                state.calulated_gpu_overhead += 1024*1024*1024*(9 - sdquanted * 1.5) # 9, 7.5, 6
            elif state.modelfile_extracted_meta[3] > 1024*1024*512: #normal sd tax
                state.calulated_gpu_overhead += 1024*1024*1024*(4.25 - sdquanted * 0.5) # 4.25, 3.75, 3.25
            if state.modelfile_extracted_meta[4] > 1024*1024*10: #whisper tax
                state.calulated_gpu_overhead += max(350*1024*1024,state.modelfile_extracted_meta[4]*1.5)
            if state.modelfile_extracted_meta[5] > 1024*1024*10: #mmproj tax
                state.calulated_gpu_overhead += max(350*1024*1024,state.modelfile_extracted_meta[5]*1.5)
            if state.modelfile_extracted_meta[6] > 1024*1024*10: #draft model tax
                state.calulated_gpu_overhead += (state.modelfile_extracted_meta[6] * 1.5)
            if state.modelfile_extracted_meta[7] > 1024*1024*10: #tts model tax
                if state.modelfile_extracted_meta[7] < 1024*1024*1024: #less than 1gb probably means outetts, which needs more vram
                    state.calulated_gpu_overhead += max(600*1024*1024, state.modelfile_extracted_meta[7] * 3)
                else:
                    state.calulated_gpu_overhead += max(600*1024*1024, (150*1024*1024 + state.modelfile_extracted_meta[7] * 1.3))
            if state.modelfile_extracted_meta[8] > 1024*1024*10: #embeddings model tax
                state.calulated_gpu_overhead += max(350*1024*1024, state.modelfile_extracted_meta[8] * 1.5)
            if state.modelfile_extracted_meta[9] > 1024*1024*10: #music llm tax
                musicoh1 = state.modelfile_extracted_meta[9] * 1.05
            if state.modelfile_extracted_meta[10] > 1024*1024*10: #music dit tax
                musicoh2 = state.modelfile_extracted_meta[10] * 1.05 + (600*1024*1024)
            if musiclowvram:
                state.calulated_gpu_overhead += max(musicoh1,musicoh2)
            else:
                state.calulated_gpu_overhead += musicoh1 + musicoh2

            mem -= state.calulated_gpu_overhead
            mem = 0 if mem < 0 else mem

            csmul = (cs/4096) if cs >= 8192 else 1.8 if cs > 4096 else 1.2 if cs > 2048 else 1.0
            ggufmeta = state.modelfile_extracted_meta[1]
            if not ggufmeta or ggufmeta[0]==0: #fail to read or no layers
                sizeperlayer = fsize*csmul*0.052
                layerlimit = int(min(200,(mem-usedmem)/sizeperlayer))
            else:
                layers = ggufmeta[0]
                headcount = ggufmeta[1]
                headkvlen = (ggufmeta[2] if ggufmeta[2] > 0 else 128)
                ratio = (mem-usedmem)/(fsize*csmul*1.6*(1.0 if bbs <= 512 else 1.2))
                if headcount > 0:
                    # rubbish random formula. apply batchsize calculations if over 512
                    fattn_discount = 1.0
                    mem1 = layers*(4 if bbs <= 512 else (bbs/128))*headkvlen*cs*fattn_discount*4*1.45
                    mem2 = layers*headcount*headkvlen*cs*fattn_discount*4*1.15
                    ratio = max(ratio,(mem - reservedmem - mem1) / (fsize + mem2))
                layerlimit = min(int(ratio*layers), (layers + 1))
        layerlimit = (0 if layerlimit<=2 else layerlimit)
        return layerlimit
    except Exception:
        return 0


def detect_memory_cu(gpumem_ignore_limit_min, gpumem_ignore_limit_max):
        FetchedCUdevices = []
        FetchedCUdeviceMem = []
        FetchedCUfreeMem = []

        AMDgpu = None
        try: # Get NVIDIA GPU names
            output = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader'], capture_output=True, text=True, check=True, encoding='utf-8', timeout=10).stdout
            FetchedCUdevices = [line.split(",")[0].strip() for line in output.splitlines()]
            FetchedCUdeviceMem = [line.split(",")[1].strip().split(" ")[0].strip() for line in output.splitlines()]
            FetchedCUfreeMem = [line.split(",")[2].strip().split(" ")[0].strip() for line in output.splitlines()]
        except Exception:
            FetchedCUdeviceMem = []
            FetchedCUfreeMem = []
            pass
        if len(FetchedCUdevices)==0:
            try: # Get AMD ROCm GPU names and VRAM from rocminfo
                output = subprocess.run(['rocminfo'], capture_output=True, text=True, check=True, encoding='utf-8', timeout=10).stdout
                device_name = None
                current_agent_is_gpu = False
                in_pool_section = False

                for line in output.splitlines(): # read through the output line by line
                    line = line.strip()
                    if line.startswith("Agent ") and "Agent" in line:
                        # Reset state for new agent
                        device_name = None
                        current_agent_is_gpu = False
                        in_pool_section = False
                    elif line.startswith("Marketing Name:"):
                        device_name = line.split(":", 1)[1].strip() # if we find a named device, temporarily save the name
                    elif line.startswith("Device Type:") and "GPU" in line and device_name is not None:
                        # if the following Device Type is a GPU (not a CPU) then add it to devices list
                        FetchedCUdevices.append(device_name)
                        current_agent_is_gpu = True
                        AMDgpu = True
                    elif line.startswith("Device Type:") and "GPU" not in line:
                        device_name = None
                        current_agent_is_gpu = False
                    elif line.startswith("Pool Info:") and current_agent_is_gpu:
                        in_pool_section = True
                    elif in_pool_section and current_agent_is_gpu and line.startswith("Segment:") and "GLOBAL" in line and "COARSE GRAINED" in line:
                        # This is the main VRAM pool for this GPU
                        continue
                    elif in_pool_section and current_agent_is_gpu and line.startswith("Size:"):
                        # Extract VRAM size in KB and convert to MB
                        size_match = re.search(r'(\d+)\(0x[0-9a-fA-F]+\)\s*KB', line)
                        if size_match:
                            vram_kb = int(size_match.group(1))
                            vram_mb = vram_kb // 1024
                            FetchedCUdeviceMem.append(str(vram_mb))
                            in_pool_section = False

                if FetchedCUdevices and FetchedCUdeviceMem:
                    print(f"Detected AMD GPU VRAM from rocminfo: {list(zip(FetchedCUdevices, FetchedCUdeviceMem))} MB")
            except Exception:
                FetchedCUdeviceMem = []
                FetchedCUfreeMem = []
                pass
        lowestcumem = 0
        lowestfreecumem = 0
        try:
            for idx in range(0,4):
                if(len(FetchedCUdevices)>idx):
                    state.CUDevicesNames[idx] = FetchedCUdevices[idx]
            for idx in range(0,4):
                if(len(FetchedCUdevices)>idx):
                    if len(FetchedCUdeviceMem)>idx:
                        dmem = (int(FetchedCUdeviceMem[idx])*1024*1024) if AMDgpu else (int(FetchedCUdeviceMem[idx])*1024*1024)
                        lowestcumem = dmem if lowestcumem==0 else (dmem if dmem<lowestcumem else lowestcumem)
                    if len(FetchedCUfreeMem)>idx:
                        dmem = (int(FetchedCUfreeMem[idx])*1024*1024)
                        lowestfreecumem = dmem if lowestfreecumem==0 else (dmem if dmem<lowestfreecumem else lowestfreecumem)
        except Exception:
            lowestcumem = 0
            lowestfreecumem = 0

        return lowestcumem, lowestfreecumem


def detect_memory_vk(gpumem_ignore_limit_min, gpumem_ignore_limit_max):

        try: # Get Vulkan names
            foundVkGPU = False
            lowestvkmem = 0
            output = subprocess.run(['vulkaninfo','--summary'], capture_output=True, text=True, check=True, encoding='utf-8', timeout=10).stdout
            devicelist = [line.split("=")[1].strip() for line in output.splitlines() if "deviceName" in line]
            devicetypes = [line.split("=")[1].strip() for line in output.splitlines() if "deviceType" in line]
            idx = 0
            for dname in devicelist:
                if idx<len(state.VKDevicesNames):
                    state.VKDevicesNames[idx] = dname
                    idx += 1
            if len(devicetypes) == len(devicelist):
                idx = 0
                for dvtype in devicetypes:
                    if idx<len(state.VKIsDGPU):
                        typeflag = (1 if dvtype=="PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" else 0)
                        state.VKIsDGPU[idx] = typeflag
                        if typeflag:
                            foundVkGPU = True
                        idx += 1

            if foundVkGPU:
                try: # Try get vulkan memory (experimental)
                    output = subprocess.run(['vulkaninfo'], capture_output=True, text=True, check=True, encoding='utf-8', timeout=10).stdout
                    devicechunks = output.split("VkPhysicalDeviceMemoryProperties")[1:]
                    gpuidx = 0
                    for chunk in devicechunks:
                        heaps = chunk.split("memoryTypes:")[0].split("memoryHeaps[")[1:]
                        for heap in heaps:  # Check all heaps, not just the first one
                            if "MEMORY_HEAP_DEVICE_LOCAL_BIT" in heap and "size" in heap:
                                match = re.search(r"size\s*=\s*(\d+)", heap)
                                if match:
                                    dmem = int(match.group(1))
                                    if dmem > gpumem_ignore_limit_min and dmem < gpumem_ignore_limit_max:
                                        lowestvkmem = dmem if lowestvkmem==0 else (dmem if dmem<lowestvkmem else lowestvkmem)
                        gpuidx += 1
                except Exception: # failed to get vulkan vram
                    pass
            return lowestvkmem
        except Exception:
            pass

        return 0


def fetch_gpu_properties(testCU,testVK,testmemory=False):
    gpumem_ignore_limit_min = 1024*1024*600 #600 mb min
    gpumem_ignore_limit_max = 1024*1024*1024*300 #300 gb max

    if testCU:
        cumem, freecumem = detect_memory_cu(gpumem_ignore_limit_min, gpumem_ignore_limit_max)
        state.MaxMemory[0] = max(cumem,state.MaxMemory[0])
        state.MaxFreeMemory[0] = max(freecumem,state.MaxFreeMemory[0])
        if testmemory:
            print(f'detected CUDA memory: {cumem/(1024*1024)} MB, {freecumem/(1024*102)} MB free')

    if testVK:
        vkmem = detect_memory_vk(gpumem_ignore_limit_min, gpumem_ignore_limit_max)
        state.MaxMemory[0] = max(vkmem,state.MaxMemory[0])
        if testmemory:
            print(f'detected Vulkan memory: {vkmem/(1024*1024)} MB')

    # Check VRAM detection after all backends have been tested
    if state.MaxMemory[0] < (1024*1024*256):
        print("Unable to detect VRAM.")

    return


def auto_set_backend_cli():
    fetch_gpu_properties(True,True)
    found_new_backend = False

    # check for avx2 and avx support
    is_oldpc_ver = "Use CPU" not in state.runopts #on oldcpu ver, default lib does not exist
    cpusupport = utils.old_cpu_check() # 0 if has avx2, 1 if has avx, 2 if has nothing
    eligible_cuda = (cpusupport<1 and not is_oldpc_ver) or (cpusupport<2 and is_oldpc_ver)
    if not eligible_cuda:
        if cpusupport==1:
            state.args.noavx2 = True
        elif cpusupport==2:
            state.args.noavx2 = True
            state.args.failsafe = True

    if eligible_cuda and state.exitcounter < 100 and state.MaxMemory[0]>3500000000 and (("Use CUDA" in state.runopts and state.CUDevicesNames[0]!="") or "Use hipBLAS (ROCm)" in state.runopts) and any(state.CUDevicesNames):
        if "Use CUDA" in state.runopts or "Use hipBLAS (ROCm)" in state.runopts:
            state.args.usecuda = ["normal","mmq"]
            print(f"Auto Selected CUDA Backend (flag={cpusupport})\n")
            found_new_backend = True
    elif state.exitcounter < 100 and (1 in state.VKIsDGPU) and ("Use Vulkan" in state.runopts or "Use Vulkan (Old CPU)" in state.runopts):
        for i in range(0,len(state.VKIsDGPU)):
            if state.VKIsDGPU[i]==1:
                state.args.usevulkan = []
                print(f"Auto Selected Vulkan Backend (flag={cpusupport})\n")
                found_new_backend = True
                break
    if not found_new_backend:
        print(f"Auto Selected Default Backend (flag={cpusupport})\n")
