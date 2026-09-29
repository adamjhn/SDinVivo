"""
Shared SD-onset extraction with the end-of-sim AP artifact removed.

Background (per PI, 2026-07-28): in run_v2*.py a cell is added to the SD
dictionary the moment its soma crosses cfg.SDThreshold (-40 mV), recording a
candidate onset. Whether it was *real* sustained depolarization vs. just an
action potential is only decided when the cell comes back *down*: intervals
that lasted <= 2.5 ms are discarded as APs. A cell that fires an AP in the last
~2.5 ms of the sim never comes back down, so it is never subjected to that
filter -- it is written as an open-ended event [onset, None] with onset pinned
at ~end-of-sim, and shows up as a spurious pile of "SD at 3000 ms".

load_filtered_onsets applies the *same 2.5 ms rule the sim uses everywhere else*
to these unresolved tail events:
  - a CLOSED interval (end is not None) is always kept -- it already passed the
    in-sim 2.5 ms AP filter;
  - an OPEN interval (end is None) is kept only if (duration - onset) > guard_ms,
    i.e. it stayed depolarized at least as long as the AP-rejection window.
A cell whose only surviving evidence is an unresolvable open tail is dropped.
"""
import glob
import json
import os


def load_filtered_onsets(data_dir, duration, guard_ms=2.5, start_ms=0.0):
    """Return {gid: earliest_surviving_onset_ms} across all cellsSD_*.json ranks.

    duration  -- cfg.duration (ms); the open-interval end is measured against it.
    guard_ms  -- min sustained time for an unresolved (open) interval to count;
                 2.5 ms matches the sim's own AP/SD cutoff in run_v2*.py.
    start_ms  -- drop any onset earlier than this (default 0.0 = keep all, so
                 existing callers are unchanged). Set >0 (e.g. 50) to exclude
                 initialization-transient / K+-seed-bolus crossings in the first
                 few ms, which are not part of the propagating-SD population and
                 can otherwise contaminate onset statistics.
    """
    onset = {}
    for fn in sorted(glob.glob(os.path.join(data_dir, "cellsSD_*.json"))):
        with open(fn) as f:
            rank_sd = json.load(f)
        for gid_str, events in rank_sd.items():
            gid = int(gid_str)
            surviving = [
                a
                for a, b in events
                if ((b is not None) or ((duration - a) > guard_ms)) and a >= start_ms
            ]
            if not surviving:
                continue
            first = min(surviving)
            onset[gid] = min(onset.get(gid, first), first)
    return onset
