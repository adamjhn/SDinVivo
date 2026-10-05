import pickle
import numpy as np
import json

# Romaro et al. 2021 (PMC8382011) discard the first 100 ms before measuring, so the
# startup transient does not enter the rate / CV / Fano statistics. The targets in
# PDNetStats.csv were produced under that convention, so it is the default here.
T_TRANSIENT = 100.0


def dict_to_latex_table(keys, values):
    # Begin LaTeX table
    latex_table = "\\begin{tabular}{|"

    # Add columns for each key in the dictionary
    for _ in keys:
        latex_table += "c|"
    latex_table += "}\n\\hline\n"

    # Add column names
    for key in keys():
        latex_table += f"{key} & "
    latex_table = latex_table[:-2]  # Remove the last '& ' from the last column name
    latex_table += "\\\\\n\\hline\n"

    # Add values as a row
    if hasattr(values[0], "__len__"):
        for v in values:
            for value in v:
                latex_table += (
                    f"{value:.2f} & " if isinstance(value, (float)) else f"{value} & "
                )
                latex_table = latex_table[
                    :-2
                ]  # Remove the last '& ' from the last value
                latex_table += "\\\\\n"
    else:
        for value in values:
            latex_table += (
                f"{value:.2f} & " if isinstance(value, (float)) else f"{value} & "
            )
            latex_table = latex_table[:-2]  # Remove the last '& ' from the last value
            latex_table += "\\\\\n"
    latex_table += "\\hline\n"

    # End LaTeX table
    latex_table += "\\end{tabular}"

    return latex_table


def networkStatsFromSimData(
    sd, pops, duration, filename=None, N=1000, seed=0, t_start=T_TRANSIENT
):
    spkt = np.array(sd["spkt"])
    spkid = np.array(sd["spkid"])
    stats = networkStats(spkt, spkid, pops, duration, N, seed=seed, t_start=t_start)
    if filename:
        json.dump(stats, open(filename, "w"))
    return stats


def networkStatsFromData(dat, filename=None, N=1000, seed=0, t_start=T_TRANSIENT,
                         pops=None):
    """Stats from a loaded netpyne _data.pkl.

    `pops` is read from dat["net"]["pops"] when the pickle carries the network (a full
    sim.saveData()).  Batch trial pickles hold only simData+simConfig, so for those the
    caller must pass `pops` explicitly (e.g. from buildFitnessArgs()).
    """
    spkt = np.array(dat["simData"]["spkt"])
    spkid = np.array(dat["simData"]["spkid"])
    if pops is None:
        if "net" not in dat or "pops" not in dat.get("net", {}):
            raise KeyError(
                "this pickle has no 'net' (keys: %s) -- pass pops= explicitly"
                % sorted(dat)
            )
        pops = {
            k: {"cellGids": v["cellGids"] if isinstance(v, dict) else v.cellGids}
            for k, v in dat["net"]["pops"].items()
        }
    duration = dat["simConfig"]["duration"]
    stats = networkStats(spkt, spkid, pops, duration, N, seed=seed, t_start=t_start)
    if filename:
        json.dump(stats, open(filename, "w"))
    return stats


def networkStatsFromSim(sim, filename=None, N=1000, seed=0, t_start=T_TRANSIENT):
    pops = {k: {"cellGids": v.cellGids} for k, v in sim.net.pops.items()}
    spkt = sim.simData["spkt"].as_numpy()
    spkid = sim.simData["spkid"].as_numpy()
    duration = sim.cfg.duration
    stats = networkStats(spkt, spkid, pops, duration, N, seed=seed, t_start=t_start)
    if filename:
        json.dump(stats, open(filename, "w"))
    return stats


def networkStats(spkt, spkid, pops, duration, Nsample=1000, seed=0, t_start=T_TRANSIENT):
    """Rate / irregularity / synchrony per population, following Romaro et al. 2021
    (Neural Comput 33:1993-2032, PMC8382011), which follows Potjans & Diesmann 2014.

    Definitions (paper eqs. 3-9):
      rate         mean over the SAMPLED neurons of spikes/second.
      irregularity CV of the inter-spike intervals computed PER NEURON, then averaged
                   over the sample.
      synchrony    var/mean of the spike-count histogram of the SAMPLE, 3 ms bins.
    All three use the same Nsample=1000 neurons per population, and the paper discards
    an initial transient (100 ms) before measuring.

    28Sep2026 -- two departures from the paper were found and fixed here:

    (1) IRREGULARITY was computed on ISIs POOLED across neurons:
            np.concatenate([isis[i] for i in sample]).std() / .mean()
        Pooling cells with different rates makes a mixture distribution whose CV is
        inflated by the between-cell rate spread, so it measured rate heterogeneity
        rather than spiking irregularity. On trial 94 it reported 1.19-2.13 against
        targets of 0.81-0.92, i.e. an apparent large failure; computed per neuron and
        averaged, the same data gives 0.85-1.06 -- on target.

    (2) SYNCHRONY used EVERY cell in the population, not the 1000-neuron sample
        (smpmap was built but only ever used for the ISIs). The Fano factor of a
        population histogram grows with N when cells are correlated, so populations
        above 1000 cells were inflated: on trial 94, L4e 63.99 (all 3506 cells) vs
        17.98 (1000 sample) against a target of 4.20.
        Also np.random.choice defaulted to replace=True, so populations SMALLER than
        1000 were sampled with duplicates. Now sampled without replacement, capped at
        the population size.

    Sampling is seeded so repeated calls on the same data agree.
    """
    L = list(pops.keys())
    rng = np.random.default_rng(seed)
    spkmap = {pop: np.asarray(pops[pop]["cellGids"]) for pop in L}
    # one sample per population, shared by all three statistics (as in the paper);
    # without replacement, and never more neurons than the population has
    smpmap = {}
    for pop in L:
        ids = spkmap[pop]
        if len(ids) == 0:
            smpmap[pop] = np.array([], dtype=int)
        else:
            k = min(Nsample, len(ids))
            smpmap[pop] = rng.choice(ids, size=k, replace=False)

    # measurement window: drop the initial transient, as the paper does
    win = (spkt >= t_start) & (spkt <= duration)
    spkt, spkid = spkt[win], spkid[win]
    T = float(duration - t_start)

    rates, sync, irr = {}, {}, {}
    bins = np.arange(t_start, duration, 3.0)      # 3 ms bins (paper)
    for pop in L:
        samp = smpmap[pop]
        if len(samp) == 0:
            rates[pop] = 0.0; sync[pop] = 0.0; irr[pop] = 0.0
            continue
        sel = np.isin(spkid, samp)                # sample only, all three stats
        ts, ids = spkt[sel], spkid[sel]

        rates[pop] = float(1e3 * len(ts) / T / len(samp))

        k = np.histogram(ts, bins=bins)[0]
        sync[pop] = float(k.var() / k.mean()) if k.mean() > 0 else 0.0

        cvs = []
        order = np.argsort(ids, kind="stable")
        ids_s, ts_s = ids[order], ts[order]
        edges = np.searchsorted(ids_s, np.unique(ids_s), side="left").tolist() + [len(ids_s)]
        for a, b in zip(edges[:-1], edges[1:]):
            d = np.diff(np.sort(ts_s[a:b]))
            if len(d) >= 2 and d.mean() > 0:      # need >=3 spikes for a usable CV
                cvs.append(d.std() / d.mean())
        irr[pop] = float(np.mean(cvs)) if cvs else 0.0

    return {"rates": rates, "synchrony": sync, "irregularity": irr}


def tailRiskStats(
    sd,
    t_eval=None,
    vBlock=-40.0,
    vMargin=15.0,
    atpLow=2.0,
    naiHigh=30.0,
    filename=None,
):
    """Tail statistics over the recorded somatic traces -- an SD-risk signature.

    Population means do NOT separate a run that goes on to SD from one that does not:
    at the 5 s window close, gen_28 (SD at 6.54 s) and gen_50 (no SD) agreed to within
    a few percent on mean ATP (2.999 vs 3.029), mean nai (16.53 vs 16.15) and the nai
    90th percentile (23.9 vs 22.3).  They differed only in the extremes -- 4 vs 1 cells
    below 2 mM ATP, 2 vs 0 cells above 30 mM nai.

    That is structural, not incidental: SD nucleates from whichever single cell tips
    into depolarisation block first, so no population-mean quantity can discriminate it.
    This function therefore reports the *tail*: how many recorded cells sit close to the
    block boundary, are ATP-depleted, or are Na-loaded, at one instant.

    `sd` is a netpyne simData dict carrying v_soma / ATPi_soma / nai_soma recordTraces.
    `t_eval` is the time in ms to evaluate at (default: the last recorded sample).
    A cell counts as near-block when v > vBlock - vMargin (default -55 mV).

    Counts are over the RECORDED cells only (typically 80 of ~13000), so they are a
    small, biased sample -- read them as an ordinal risk index, not an incidence rate.
    """
    t = np.array(sd["t"], dtype=float)
    i = len(t) - 1 if t_eval is None else int(np.argmin(np.abs(t - t_eval)))

    def col(key):
        if key not in sd:
            return None
        return np.array([v[i] for v in sd[key].values()], dtype=float)

    v, atp, nai = col("v_soma"), col("ATPi_soma"), col("nai_soma")
    out = {"t": float(t[i]), "n_cells": 0 if v is None else int(v.size)}

    if v is not None:
        out.update(
            v_p90=float(np.percentile(v, 90)),
            v_max=float(v.max()),
            n_near_block=int((v > vBlock - vMargin).sum()),
            n_blocked=int((v > vBlock).sum()),
        )
    if atp is not None:
        out.update(
            atp_mean=float(atp.mean()),
            atp_p10=float(np.percentile(atp, 10)),
            atp_min=float(atp.min()),
            n_atp_low=int((atp < atpLow).sum()),
        )
    if nai is not None:
        out.update(
            nai_mean=float(nai.mean()),
            nai_p90=float(np.percentile(nai, 90)),
            nai_max=float(nai.max()),
            n_nai_high=int((nai > naiHigh).sum()),
        )

    # Single ordinal index: how much of the recorded population is in the bad tail.
    n = max(out["n_cells"], 1)
    out["tail_index"] = (
        out.get("n_atp_low", 0) + out.get("n_nai_high", 0) + out.get("n_near_block", 0)
    ) / n

    if filename:
        json.dump(out, open(filename, "w"))
    return out
