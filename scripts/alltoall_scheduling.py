#!/usr/bin/env python3
import numpy as np
import sys

def can_fill_loc(schedule, i, j, val):
    N = schedule.shape[0]
    for k in range(N):
        if schedule[i,k] == val or schedule[k,j] == val:
            return False
    return True


def compute_schedule_impl(schedule, i, j):
    N = schedule.shape[0]
    if i == N:
        return False
    if j == N:
        return False
    for val in [(k+i)%N for k in range(N)]:
        if can_fill_loc(schedule, i, j, val):
            schedule[i,j] = val
            schedule[j,i] = val
            next_i = i + 1
            next_j = j
            if next_i == N:
                next_i = j + 1
                next_j = j + 1
            if next_j == N:
                print(schedule)
                return True
            if compute_schedule_impl(schedule, next_i, next_j):
                return True
            else:
                schedule[i,j] = -1
                schedule[j,i] = -1
                continue
    return False


def compute_schedule(N):
    schedule = -np.ones((N, N))
    return compute_schedule_impl(schedule, 0, 0)


if __name__ == '__main__':
    N_max = int(sys.argv[1])
    for N in range(1, N_max+1):
        compute_schedule(N)
