import sys
import time
import random
import threading
from datetime import datetime

from . import state
from .utils import get_my_epurl, make_url_request, print_with_time


def run_horde_worker(args, api_key, worker_name, worker_id, parallel_batching_threads):
    epurl = get_my_epurl()

    def submit_completed_generation(url, jobid, sessionstart, submit_dict):
        reply = make_url_request_horde(url, submit_dict)
        if not reply:
            state.punishcounter += 1
            print_with_time(f"Worker {worker_id} - Error, Job submit failed.")
        else:
            reward = reply["reward"]
            state.session_kudos_earned += reward
            state.session_jobs += 1
            curtime = datetime.now()
            elapsedtime = curtime - sessionstart
            hrs_float = elapsedtime.total_seconds() / 3600
            hrs = int(hrs_float)
            mins = elapsedtime.seconds // 60 % 60
            secs = elapsedtime.seconds % 60
            elapsedtimestr = f"{hrs:03d}h:{mins:02d}m:{secs:02d}s"
            earnrate = state.session_kudos_earned / hrs_float
            jobrate = state.session_jobs / hrs_float
            jobcost = state.session_kudos_earned / state.session_jobs
            print_with_time(f'Worker {worker_id} - Submitted {jobid} and earned {reward:.0f} kudos\n[Total:{state.session_kudos_earned:.0f} kudos, Time:{elapsedtimestr}, Jobs:{state.session_jobs}, EarnRate:{earnrate:.2f} kudos/hr, JobRate:{jobrate:.2f} jobs/hr, JobCost:{jobcost:.2f} kudos/job]')
            state.rewardcounter += 1
            if state.rewardcounter > 50:
                state.rewardcounter = 0
                if state.exitcounter > 1:
                    state.exitcounter -= 1

    def make_url_request_horde(url, data, method='POST', addmykey=False):
        headers = {"apikey": api_key, 'User-Agent': 'KoboldCppEmbeddedWorkerV2', 'Client-Agent': 'KoboldCppEmbedWorker:2'}
        if addmykey and state.password != "":
            headers["Authorization"] = f"Bearer {state.password}"
        ret = make_url_request(url, data, method, headers)
        if not ret:
            print(f"Worker {worker_id} - Make sure your Horde API key and worker name is valid!")
        return ret

    current_id = None
    current_payload = None
    current_generation = None
    state.session_starttime = datetime.now()
    sleepy_counter = 0  # if this exceeds a value, worker becomes sleepy (slower)
    state.exitcounter = 0
    print(f"===\nEmbedded Horde Worker '{worker_name}' Starting...\n(To use your own Horde Bridge/Scribe worker instead, don't set your API key)\n")
    BRIDGE_AGENT = "KoboldCppEmbedWorker:2:https://github.com/LostRuins/koboldcpp"
    cluster = "https://aihorde.net"
    while state.exitcounter < 10:
        time.sleep(3)
        readygo = make_url_request_horde(f'{epurl}/api/v1/info/version', None, 'GET', addmykey=True)
        if readygo:
            print_with_time(f"Worker {worker_id} - Embedded Horde Worker '{worker_name}' is started.")
            break

    while state.exitcounter < 10:
        currentjob_attempts = 0
        current_generation = None

        if state.punishcounter >= 5:
            state.punishcounter = 0
            state.exitcounter += 1
            if state.exitcounter < 10:
                penaltytime = (2 ** state.exitcounter)
                print_with_time(f"Horde Worker Paused for {penaltytime} min - Too many errors. It will resume automatically, but you should restart it.")
                print_with_time("Caution: Too many failed jobs may lead to entering maintenance mode.")
                time.sleep(60 * penaltytime)
                print_with_time(f"Worker {worker_id} - Horde Worker Resumed")
            else:
                print_with_time(f"Worker {worker_id} - Horde Worker Exit limit reached, too many errors.")

        sec_since_non_horde = time.time() - state.last_non_horde_req_time
        no_recent_local_usage = sec_since_non_horde > 20
        if not no_recent_local_usage:
            # print_with_time(f"Recent Local Usage - Horde Worker Waiting...")
            time.sleep(1)
            continue

        # first, make sure we are not generating (queue is empty)
        if state.modelbusy.locked():
            time.sleep(0.2)
            continue

        # pop new request
        gen_dict = {
            "name": worker_name,
            "models": [state.friendlymodelname],
            "max_length": state.maxhordelen,
            "max_context_length": min(state.maxctx, (state.maxctx if state.maxhordectx == 0 else state.maxhordectx)),
            "priority_usernames": [],
            "softprompts": [],
            "bridge_agent": BRIDGE_AGENT,
        }
        if parallel_batching_threads > 1:
            gen_dict["threads"] = parallel_batching_threads
        pop = make_url_request_horde(f'{cluster}/api/v2/generate/text/pop', gen_dict)
        if not pop:
            state.punishcounter += 1
            print_with_time(f"Worker {worker_id} - Failed to fetch job from {cluster}. Waiting 10 seconds...")
            time.sleep(10)
            continue
        if not pop["id"]:
            slp = (1 if sleepy_counter < 10 else (2 if sleepy_counter < 25 else 3))
            time.sleep(slp)
            sleepy_counter += 1
            if sleepy_counter == 20:
                print_with_time(f"Worker {worker_id} - No recent jobs, entering low power mode...")
            continue

        sleepy_counter = 0
        current_id = pop['id']
        current_payload = pop['payload']
        print("")  # empty newline
        print_with_time(f"Job {current_id} received from {cluster} for {current_payload.get('max_length', 0)} tokens and {current_payload.get('max_context_length', 0)} max context. Starting generation...")

        # do gen
        while state.exitcounter < 10:
            if parallel_batching_threads > 1 or not state.modelbusy.locked():
                # horde gets a genkey to avoid KCPP overlap
                current_payload['genkey'] = f"HORDEREQ_{random.randint(100, 999)}"
                current_generation = make_url_request_horde(f'{epurl}/api/v1/generate', current_payload, method='POST', addmykey=True)
                if current_generation:
                    break
                else:
                    currentjob_attempts += 1
                    if currentjob_attempts > 5:
                        break

            print_with_time(f"Worker {worker_id} - Server Busy - Not ready to generate...")
            time.sleep(5)

        # submit reply
        print("")  # empty newline
        if current_generation:
            submit_dict = {
                "id": current_id,
                "generation": current_generation["results"][0]["text"],
                "state": "ok"
            }
            submiturl = cluster + '/api/v2/generate/text/submit'
            submit_thread = threading.Thread(target=submit_completed_generation, args=(submiturl, current_id, state.session_starttime, submit_dict))
            submit_thread.start()  # submit job in new thread so nothing is waiting
        else:
            print_with_time(f"Worker {worker_id} - Error, Abandoned current job due to errors. Getting new job.")
        current_id = None
        current_payload = None
        time.sleep(0.1)

    if state.exitcounter < 100:
        print_with_time(f"Worker {worker_id} - Horde Worker Shutdown - Too many errors.")
    else:
        print_with_time(f"Worker {worker_id} - Horde Worker Shutdown - Server Closing.")
    state.exitcounter = 999
    time.sleep(3)
    sys.exit(2)
