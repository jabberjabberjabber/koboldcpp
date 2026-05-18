import os
import re
import sys
import time
import struct
import shutil
import threading
import subprocess
from datetime import datetime

from . import state


def has_valid_model():
    return state.args.model_param or state.args.sdmodel or state.args.whispermodel or state.args.ttsmodel or state.args.embeddingsmodel or state.args.musicdiffusion or state.args.musicllm or state.args.mcpfile or state.args.nomodel

def unpack_to_dir(destpath = ""):
    srcpath = state.getdirpath()
    cliunpack = False if destpath == "" else True
    print("Attempt to unpack KoboldCpp into directory...")

    if not cliunpack:
        from tkinter import messagebox
        from .zenity_gui import zentk_askdirectory
        destpath = zentk_askdirectory(title='Select an empty folder to unpack KoboldCpp')
        if not destpath:
            return

    if not os.path.isdir(destpath):
        os.makedirs(destpath)

    if os.path.isdir(srcpath) and os.path.isdir(destpath) and not os.listdir(destpath):
        try:
            if cliunpack:
                print(f"KoboldCpp will be extracted to {destpath}\nThis process may take several seconds to complete.")
            else:
                messagebox.showinfo("Unpack Starting", f"KoboldCpp will be extracted to {destpath}\nThis process may take several seconds to complete.")
            pyds_dir = os.path.join(destpath, 'pyds')
            using_pyinstaller_6 = False
            try:
                import pkg_resources
                piver = pkg_resources.get_distribution("pyinstaller").version
                print(f"PyInstaller Version: {piver}")
                if piver.startswith("6."):
                    using_pyinstaller_6 = True
                    os.makedirs(os.path.join(destpath, "_internal"), exist_ok=True)
                    pyds_dir = os.path.join(os.path.join(destpath, "_internal"), 'pyds')
            except Exception:
                pass
            os.makedirs(pyds_dir, exist_ok=True)
            for item in os.listdir(srcpath):
                s = os.path.join(srcpath, item)
                d = os.path.join(destpath, item)
                d2 = d  #this will be modified for pyinstaller 6 and unmodified for pyinstaller 5
                if using_pyinstaller_6:
                    d2 = os.path.join(os.path.join(destpath, "_internal"), item)
                if using_pyinstaller_6 and item.startswith('koboldcpp-launcher'):  # Move koboldcpp-launcher to its intended location
                    shutil.copy2(s, d)
                    continue
                if item.endswith('.pyd'):  # relocate pyds files to subdirectory
                    pyd = os.path.join(pyds_dir, item)
                    shutil.copy2(s, pyd)
                    continue
                if os.path.isdir(s):
                    shutil.copytree(s, d2, False, None)
                else:
                    shutil.copy2(s, d2)
            if cliunpack:
                print(f"KoboldCpp successfully extracted to {destpath}")
            else:
                messagebox.showinfo("KoboldCpp Unpack Success", f"KoboldCpp successfully extracted to {destpath}")
        except Exception as e:
            if cliunpack:
                print(f"An error occurred while unpacking: {e}")
            else:
                messagebox.showerror("Error", f"An error occurred while unpacking: {e}")
    else:
        if cliunpack:
            print("The target folder is not empty or invalid. Please select an empty folder.")
        else:
            messagebox.showwarning("Invalid Selection", "The target folder is not empty or invalid. Please select an empty folder.")

def exit_with_error(code, message, title="Error"):
    print("")
    time.sleep(1)
    if state.using_gui_launcher:
        from .gui import show_gui_msgbox
        show_gui_msgbox(title, message)
    else:
        print(message, flush=True)
    time.sleep(2)
    sys.exit(code)

def dump_gguf_metadata(file_path): #if you're gonna copy this into your own project at least credit concedo
    chunk_size = 1024*1024*20  # read first 20mb of file
    try:
        data = None
        fptr = 0
        dt_table = ["u8","i8","u16","i16","u32","i32","f32","bool","str","arr","u64","i64","f64"] #13 types, else error
        tt_table = ["f32","f16","q4_0","q4_1","q4_2","q4_3","q5_0","q5_1","q8_0","q8_1","q2_k","q3_k","q4_k","q5_k","q6_k","q8_k","iq2_xxs","iq2_xs","iq3_xxs","iq1_s","iq4_nl","iq3_s","iq2_s","iq4_xs","i8","i16","i32","i64","f64","iq1_m","bf16","q4_0_4_4","q4_0_4_8","q4_0_8_8","tq1_0","tq2_0","iq4_nl_4_4","iq4_nl_4_8","iq4_nl_8_8","mxfp4","nvfp4","q1_0","unknown","unknown","unknown","unknown"]
        def read_data(datatype):
            nonlocal fptr, data, dt_table
            if datatype=="u32":
                val_bytes = data[fptr:fptr + 4]
                val = struct.unpack('<I', val_bytes)[0]
                fptr += 4
                return val
            if datatype=="u64":
                val_bytes = data[fptr:fptr + 8]
                val = struct.unpack('<Q', val_bytes)[0]
                fptr += 8
                return val
            if datatype=="i32":
                val_bytes = data[fptr:fptr + 4]
                val = struct.unpack('<i', val_bytes)[0]
                fptr += 4
                return val
            if datatype=="bool":
                val_bytes = data[fptr:fptr + 1]
                val = struct.unpack('<B', val_bytes)[0]
                fptr += 1
                return val
            if datatype=="f32":
                val_bytes = data[fptr:fptr + 4]
                val = struct.unpack('<f', val_bytes)[0]
                fptr += 4
                return val
            if datatype=="str":
                val_bytes = data[fptr:fptr + 8]
                str_len = struct.unpack('<Q', val_bytes)[0]
                fptr += 8
                val_bytes = data[fptr:fptr + str_len]
                str_val = val_bytes.split(b'\0', 1)[0].decode('utf-8')
                fptr += str_len
                return str_val
            if datatype == "u16":
                val_bytes = data[fptr:fptr + 2]
                val = struct.unpack('<H', val_bytes)[0]
                fptr += 2
                return val
            if datatype == "i16":
                val_bytes = data[fptr:fptr + 2]
                val = struct.unpack('<h', val_bytes)[0]
                fptr += 2
                return val
            if datatype == "u8":
                val_bytes = data[fptr:fptr + 1]
                val = struct.unpack('<B', val_bytes)[0]
                fptr += 1
                return val
            if datatype == "i8":
                val_bytes = data[fptr:fptr + 1]
                val = struct.unpack('<b', val_bytes)[0]
                fptr += 1
                return val
            if datatype=="arr":
                val_bytes = data[fptr:fptr + 4]
                arr_type = struct.unpack('<I', val_bytes)[0]
                fptr += 4
                val_bytes = data[fptr:fptr + 8]
                arr_elems = struct.unpack('<Q', val_bytes)[0]
                fptr += 8
                arr_vals = []
                for i in range(arr_elems):
                    dt_translated = dt_table[arr_type]
                    arr_val = read_data(dt_translated)
                    arr_vals.append(arr_val)
                return arr_vals
            print(f"Unknown Datatype: {datatype}")
            return

        fsize = os.path.getsize(file_path)
        if fsize < 512: #ignore files under file size limit
            print("This GGUF file is too small to analyze. Please ensure it is valid.")
            return
        with open(file_path, 'rb') as f:
            file_header = f.read(4)
            if file_header != b'GGUF': #file is not GGUF
                print(f"File does not seem to be a GGUF: {file_header}")
                return
            data = f.read(chunk_size)
            read_ver = read_data("u32")
            if read_ver < 2:
                print(f"This GGUF file is too old. Version detected: {read_ver}")
                return
            read_tensorcount = read_data("u64")
            read_kvcount = read_data("u64")
            print(f"*** GGUF FILE METADATA ***\nGGUF.version = {read_ver}\nGGUF.tensor_count = {read_tensorcount}\nGGUF.kv_count = {read_kvcount}")
            for kn in range(read_kvcount):
                curr_key = read_data("str")
                curr_datatype = read_data("u32")
                dt_translated = dt_table[curr_datatype]
                curr_val = read_data(dt_translated)
                if dt_translated=="arr":
                    print(f"{dt_translated}: {curr_key} = [{len(curr_val)}]")
                elif dt_translated=="str":
                    print(f"{dt_translated}: {curr_key} = {curr_val[:256]}")
                else:
                    print(f"{dt_translated}: {curr_key} = {curr_val}")
            print("\n*** GGUF TENSOR INFO ***")
            for kn in range(read_tensorcount):
                tensor_name = read_data("str")
                dims = read_data("u32")
                dim_val_str = "["
                for d in range(dims):
                    dim_val = read_data("u64")
                    dim_val_str += f"{'' if d==0 else ', '}{dim_val}"
                dim_val_str += "]"
                tensor_type = read_data("u32")
                read_data("u64") # tensor_offset not used
                tensor_type_str = tt_table[tensor_type]
                print(f"{kn:<3}: {tensor_type_str:<8} | {tensor_name:<30} | {dim_val_str}")
            print(f"Metadata and TensorInfo Bytes: {fptr}")
    except Exception as e:
        print(f"Error Analyzing File: {e}")
        return

def read_gguf_metadata(file_path):
    chunk_size = 16384  # read only first 16kb of file
    try:
        def read_gguf_key(keyname,data,maxval):
            keylen = len(keyname)
            index = data.find(keyname)  # Search for the magic number, Read 2 chunks of 4 byte numbers
            if index != -1 and index + keylen + 8 <= chunk_size:
                start_index = index + keylen
                first_value_bytes = data[start_index:start_index + 4]
                second_value_bytes = data[start_index + 4:start_index + 8]
                # Unpack each 4 bytes as an unsigned int32 in little-endian format
                value1 = struct.unpack('<I', first_value_bytes)[0] #4 means its a uint32
                value2 = struct.unpack('<I', second_value_bytes)[0]
                if value1 == 4 and value2 > 0 and value2 <= maxval:
                    return value2 #contains the desired value
                return 0
            else:
                return 0 #not found

        fsize = os.path.getsize(file_path)
        if fsize < (chunk_size+256): #ignore files under 16kb
            return None
        with open(file_path, 'rb') as f:
            file_header = f.read(4)
            if file_header != b'GGUF': #file is not GGUF
                return None
            data = f.read(chunk_size)
            layercount = read_gguf_key(b'.block_count',data,512)
            head_count_kv = read_gguf_key(b'.attention.head_count_kv',data,8192)
            key_length = read_gguf_key(b'.attention.key_length',data,8192)
            val_length = read_gguf_key(b'.attention.value_length',data,8192)
            return [layercount,head_count_kv, max(key_length,val_length)]
    except Exception:
        return None

def extract_modelfile_params(filepath,sdfilepath,whisperfilepath,mmprojfilepath,draftmodelpath,ttsmodelpath,embdmodelpath,musicllmpath,musicditpath):
    state.modelfile_extracted_meta = None
    sdfsize = 0
    whisperfsize = 0
    mmprojsize = 0
    draftmodelsize = 0
    ttsmodelsize = 0
    embdmodelsize = 0
    musicllmsize = 0
    musicditsize = 0
    if sdfilepath and os.path.exists(sdfilepath):
        sdfsize = os.path.getsize(sdfilepath)
    if whisperfilepath and os.path.exists(whisperfilepath):
        whisperfsize = os.path.getsize(whisperfilepath)
    if mmprojfilepath and os.path.exists(mmprojfilepath):
        mmprojsize = os.path.getsize(mmprojfilepath)
    if draftmodelpath and os.path.exists(draftmodelpath):
        draftmodelsize = os.path.getsize(draftmodelpath)
    if ttsmodelpath and os.path.exists(ttsmodelpath):
        ttsmodelsize = os.path.getsize(ttsmodelpath)
    if embdmodelpath and os.path.exists(embdmodelpath):
        embdmodelsize = os.path.getsize(embdmodelpath)
    if musicllmpath and os.path.exists(musicllmpath):
        musicllmsize = os.path.getsize(musicllmpath)
    if musicditpath and os.path.exists(musicditpath):
        musicditsize = os.path.getsize(musicditpath)
    if filepath and os.path.exists(filepath):
        try:
            fsize = os.path.getsize(filepath)
            if fsize>10000000: #dont bother with models < 10mb as they are probably bad
                ggufmeta = read_gguf_metadata(filepath)
                state.modelfile_extracted_meta = [filepath,ggufmeta,fsize,sdfsize,whisperfsize,mmprojsize,draftmodelsize,ttsmodelsize,embdmodelsize,musicllmsize,musicditsize] #extract done. note that meta may be null
        except Exception:
            state.modelfile_extracted_meta = None

def delete_old_pyinstaller():
    try:
        base_path = sys._MEIPASS
    except Exception:
        return # not running from pyinstaller
    if not base_path:
        return

    selfdirpath = os.path.abspath(base_path)
    temp_parentdir_path = os.path.abspath(os.path.join(base_path, '..'))
    for dirname in os.listdir(temp_parentdir_path):
        absdirpath = os.path.abspath(os.path.join(temp_parentdir_path, dirname))
        if os.path.isdir(absdirpath) and os.path.basename(absdirpath).startswith('_MEI'): #only delete kobold pyinstallers
            if absdirpath!=selfdirpath and (time.time() - os.path.getctime(absdirpath)) > 14400: # remove if older than 4 hours
                kobold_itemcheck1 = os.path.join(absdirpath, 'koboldcpp_default.dll')
                kobold_itemcheck2 = os.path.join(absdirpath, 'koboldcpp_default.so')
                kobold_itemcheck3 = os.path.join(absdirpath, 'koboldcpp.py')
                kobold_itemcheck4 = os.path.join(absdirpath, 'cublasLt64_11.dll')
                kobold_itemcheck5 = os.path.join(absdirpath, 'cublas64_11.dll')
                if os.path.exists(kobold_itemcheck1) or os.path.exists(kobold_itemcheck2) or os.path.exists(kobold_itemcheck3) or (os.path.exists(kobold_itemcheck4) and os.path.exists(kobold_itemcheck5)):
                    try:
                        shutil.rmtree(absdirpath)
                        print(f"Deleted orphaned pyinstaller dir: {absdirpath}")
                    except Exception as e:
                        print(f"Error deleting orphaned pyinstaller dir: {absdirpath}: {e}")

def downloader_internal(input_url, output_filename, capture_output, min_file_size=64): # 64 bytes required by default
    download_dir_path = state.args.downloaddir
    if "https://huggingface.co/" in input_url and "/blob/main/" in input_url:
        input_url = input_url.replace("/blob/main/", "/resolve/main/")
    if download_dir_path:
        download_dir_path = os.path.abspath(download_dir_path)
        os.makedirs(download_dir_path, exist_ok=True)
    if output_filename != "auto" and download_dir_path and not os.path.isabs(output_filename):
        output_filename = os.path.join(download_dir_path, output_filename)
    if output_filename == "auto":
        filename = os.path.basename(input_url).split('?')[0].split('#')[0]
        if download_dir_path:
            output_filename = os.path.join(download_dir_path, filename)
        else:
            cwd = os.getcwd()
            non_writable = False
            if os.name == "nt":
                parts = [p.lower() for p in os.path.normpath(cwd).split(os.sep)]
                if "windows" in parts and ("system32" in parts or "syswow64" in parts):
                    non_writable = True
            if not non_writable:
                output_filename = filename
            else:
                exe_dir = os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__)
                output_filename = os.path.join(exe_dir, filename)
    incomplete_dl_exist = (os.path.exists(output_filename+".aria2") and os.path.getsize(output_filename+".aria2") > 16)
    if os.path.exists(output_filename) and os.path.getsize(output_filename) > min_file_size and not incomplete_dl_exist:
        print(f"{output_filename} already exists, using existing file.")
        return output_filename
    print(f"Downloading {input_url}", flush=True)

    dl_success = False
    out_dir = os.path.dirname(os.path.abspath(output_filename)) or os.getcwd()
    out_name = os.path.basename(output_filename)
    try:
        if os.name == 'nt':
            basepath = state.getdirpath()
            a2cexe = os.path.join(basepath, "aria2c-win.exe")
            if os.path.exists(a2cexe):  # on windows try using embedded aria2c
                rc = subprocess.run([
                        a2cexe, "-x", "16", "-s", "16",
                        "--summary-interval=15", "--console-log-level=error", "--log-level=error",
                        "--download-result=default", "--continue=true", "--allow-overwrite=true",
                        "--file-allocation=none", "--max-tries=3",
                        "-d", out_dir, "-o", out_name, input_url
                    ], capture_output=capture_output, text=True, check=True, encoding='utf-8')
                dl_success = (rc.returncode == 0 and os.path.exists(output_filename) and os.path.getsize(output_filename) > min_file_size)
    except subprocess.CalledProcessError as e:
        print(f"aria2c-win failed: {e}")

    try:
        if not dl_success and shutil.which("aria2c") is not None:
            rc = subprocess.run([
                    "aria2c", "-x", "16", "-s", "16",
                    "--summary-interval=15", "--console-log-level=error", "--log-level=error",
                    "--download-result=default", "--allow-overwrite=true",
                    "--file-allocation=none", "--max-tries=3",
                    "-d", out_dir, "-o", out_name, input_url
                ], capture_output=capture_output, text=True, check=True, encoding='utf-8')
            dl_success = (rc.returncode == 0 and os.path.exists(output_filename) and os.path.getsize(output_filename) > min_file_size)
    except subprocess.CalledProcessError as e:
        print(f"aria2c failed: {e}")

    try:
        if not dl_success and shutil.which("curl") is not None:
            rc = subprocess.run(["curl", "-fLo", output_filename, input_url],
                capture_output=capture_output, text=True, check=True, encoding="utf-8")
            dl_success = (rc.returncode == 0 and os.path.exists(output_filename) and os.path.getsize(output_filename) > min_file_size)
    except subprocess.CalledProcessError as e:
        print(f"curl failed: {e}")

    try:
        if not dl_success and shutil.which("wget") is not None:
            rc = subprocess.run(["wget", "-O", output_filename, input_url],
                capture_output=capture_output, text=True, check=True, encoding="utf-8")
            dl_success = (rc.returncode == 0 and os.path.exists(output_filename) and os.path.getsize(output_filename) > min_file_size)
    except subprocess.CalledProcessError as e:
        print(f"wget failed: {e}")

    if not dl_success:
        print("Could not find suitable download software, or all download methods failed. Please install aria2, curl, or wget.")
        return None

    return output_filename


def download_model_from_url(url, permitted_types=[".gguf",".safetensors", ".ggml", ".bin"], min_file_size=64,handle_multipart=False):
    if url and url!="":
        if url.endswith("?download=true"):
            url = url.replace("?download=true","")
        end_ext_ok = False
        for t in permitted_types:
            if url.endswith(t):
                end_ext_ok = True
                break
        if ((url.startswith("http://") or url.startswith("https://")) and end_ext_ok):
            dlfile = downloader_internal(url, "auto", False, min_file_size)
            if handle_multipart and "-00001-of-00" in url: #handle multipart files up to 9 parts
                match = re.search(r'-(\d{5})-of-(\d{5})\.', url)
                if match:
                    total_parts = int(match.group(2))
                    if total_parts > 1 and total_parts <= 999:
                        current_part = 1
                        base_url = url
                        for part_num in range(current_part + 1, total_parts + 1):
                            part_str = f"-{part_num:05d}-of-{total_parts:05d}"
                            new_url = re.sub(r'-(\d{5})-of-(\d{5})', part_str, base_url)
                            downloader_internal(new_url, "auto", False, min_file_size)
            return dlfile
    return None

def analyze_gguf_model(args,filename):
    try:
        stime = datetime.now()
        dump_gguf_metadata(filename)
        atime = (datetime.now() - stime).total_seconds()
        print(f"---\nAnalyzing completed in {atime:.2f}s.\n---",flush=True)
    except Exception as e:
        print(f"Cannot Analyze File: {e}")
    return

def analyze_gguf_model_wrapper(filename=""):
    from .zenity_gui import zentk_askopenfilename
    if not filename or filename=="":
        try:
            filename = zentk_askopenfilename(title="Select GGUF to analyze")
        except Exception as e:
            print(f"Cannot select file to analyze: {e}")
    if not filename or filename=="" or not os.path.exists(filename):
        print("Selected GGUF file not found. Please select a valid GGUF file to analyze.")
        return
    print("---")
    print(f"Analyzing {filename}, please wait...\n---",flush=True)
    dumpthread = threading.Thread(target=analyze_gguf_model, args=(state.args,filename))
    dumpthread.start()
