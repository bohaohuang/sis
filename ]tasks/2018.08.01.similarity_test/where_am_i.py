import subprocess
import numpy as np


def get_input():
    return subprocess.run('squeue | grep GPU-share', shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')


def cnt_first_occurrence(stats_list, sub_str, extra_str=''):
    for cnt, line in enumerate(stats_list):
        if sub_str in line and extra_str in line:
            return cnt + 1


def cnt_last_occurrence(stats_list, sub_str, extra_str=''):
    pos = 0
    for cnt, line in enumerate(stats_list):
        if sub_str in line and extra_str in line:
            pos = cnt + 1
    return pos


def get_all_occurrence(stats_list, sub_str, extra_str=''):
    strs = []
    for cnt, line in enumerate(stats_list):
        if sub_str in line and extra_str in line:
            strs.append(line)
    return strs


def get_time(line):
    time_str = line.strip().split()[5]
    time_str = ''.join([a for a in time_str if a.isdigit()])
    return int(time_str)


def sort_by_time(lines):
    time_record = []
    for line in lines:
        time_record.append(get_time(line))
    sort_idx = np.argsort(time_record)[::-1]
    return [lines[a] for a in sort_idx]


if __name__ == '__main__':
    stats = get_input().split('\n')
    my_pos = cnt_last_occurrence(stats, 'bohaohua', 'PD')
    first_pos = cnt_first_occurrence(stats, 'gpu')
    print('Ugh! {} devils infront of me ...'.format(first_pos-my_pos))

    print('Running stats sort by time:')
    running_lines = get_all_occurrence(stats, 'gpu')
    sort_lines = sort_by_time(running_lines)
    for line in sort_lines:
        print(line)
