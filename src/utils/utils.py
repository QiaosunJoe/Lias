#!/usr/bin/env python3
# coding: utf-8
import os
import numpy as np
import subprocess


def parse_perf_output(file_path):
    """Parse perf stat output file"""
    event_dict = {}
    time_step = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines[3:]:
        parts = line.strip().split()

        if len(parts) < 4 or parts[0] == '#':
            continue

        time = int(float(parts[0]))
        count = parts[1].replace(',', '')

        if count == '<not':
            count = 0
            event_name = parts[3]
            if event_name == 'Bytes':
                event_name = parts[4]
        else:
            event_name = parts[2]
            if event_name == 'MiB' or event_name == 'msec' or event_name == 'Bytes':
                event_name = parts[3]

        if event_name not in event_dict:
            event_dict[event_name] = []
            time_step[event_name] = 1

        if time == time_step[event_name]:
            event_dict[event_name].append(round(float(count)))
            time_step[event_name] += 1
        else:
            num = time - time_step[event_name]
            for n in range(num):
                event_dict[event_name].append(event_dict[event_name][-1])
            event_dict[event_name].append(round(float(count)))
            time_step[event_name] = time + 1

    return event_dict


EVENT_GROUP_FILES = [
    'hardware_cache_events.txt',
    'hardware_events.txt',
    'software_events.txt',
    'kernel_pmu_events.txt',
    'memory_events.txt',
    'pipeline_events.txt',
    'cache_specific_events.txt',
    'floating_point_events.txt',
    'frontend_events.txt',
    'virtual_memory_events.txt',
]


class Lat(object):
    def __init__(self, fileName):
        f = open(fileName, 'rb')
        a = np.fromfile(f, dtype=np.uint64)
        self.reqTimes = a.reshape((a.shape[0] // 3, 3))
        f.close()

    def parseQueueTimes(self):
        return self.reqTimes[:, 0]

    def parseSvcTimes(self):
        return self.reqTimes[:, 1]

    def parseSojournTimes(self):
        return self.reqTimes[:, 2]


def getLatPct(latsFile,file_path):
    assert os.path.exists(latsFile)

    latsObj = Lat(latsFile)

    qTimes = [l/1e6 for l in latsObj.parseQueueTimes()]
    svcTimes = [l/1e6 for l in latsObj.parseSvcTimes()]
    sjrnTimes = [l/1e6 for l in latsObj.parseSojournTimes()]
    
    f = open(f'{file_path}','w')

    f.write('%12s | %12s | %12s\n\n' \
            % ('QueueTimes', 'ServiceTimes', 'SojournTimes'))

    for (q, svc, sjrn) in zip(qTimes, svcTimes, sjrnTimes):
        f.write("%12s | %12s | %12s\n" \
                % ('%.3f' % q, '%%.3f' % svc, '%.3f' % sjrn))
    f.close()


def reset_COS():
    command = "sudo pqos --iface=os -R"
    result = subprocess.Popen(command, shell= True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    res, error = result.communicate()
    res = res.decode('utf-8').split("\n")
    return res, error


def extract_data_log(file_path):
    lats = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

        for i in range(3, len(lines), 5):
            number = int(lines[i].strip())
            number = round(number / 1000000, 4)
            lats.append(number)
    return lats
