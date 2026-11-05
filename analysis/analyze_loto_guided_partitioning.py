#!/usr/bin/env python3
"""Exact LOTO-guided partitioning of the frozen 14-task VenusPheno panel.

This is an analysis-only script.  It solves the requested signed, directed
within-group objective exactly using a subset dynamic program; it never loads
models, changes source experiments, or runs a GPU job.
"""
from __future__ import annotations

import itertools
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parent
INTERACTION_ROOT = ROOT.parent / "20261018_loto_transfer_interaction"
INPUT = INTERACTION_ROOT / "outputs"
HISTORICAL = INTERACTION_ROOT / "inputs" / "historical_grouped_mtl"
OUT = ROOT / "outputs"
PLOTS = OUT / "plots"
RNG = np.random.default_rng(20261018)
SEEDS = (42, 43, 44)
MIN_SIZE, MAX_SIZE, N_RANDOM = 3, 6, 10_000

SHORT = {
    "motility": "Motility", "sporulation": "Sporulation", "gram_stain": "Gram stain",
    "metabolism__aerobic": "Aerobic", "metabolism__anaerobic": "Anaerobic",
    "metabolism__facultative": "Facultative", "cell_shape__bacillus": "Bacillus",
    "cell_shape__coccus": "Coccus", "carb_citrate": "Citrate",
    "carb_glucose": "Glucose", "carb_glycerol": "Glycerol", "carb_lactose": "Lactose",
    "carb_maltose": "Maltose", "carb_sucrose": "Sucrose",
}
DERIVED_FAMILIES = {
    "cell_shape": {"cell_shape__bacillus", "cell_shape__coccus"},
    "metabolism_oxygen": {"metabolism__aerobic", "metabolism__anaerobic", "metabolism__facultative"},
}
TRANSFER_GROUPS = [
    ["metabolism__anaerobic", "cell_shape__bacillus", "cell_shape__coccus"],
    ["gram_stain", "carb_citrate", "carb_glucose", "carb_glycerol", "carb_lactose", "carb_sucrose"],
    ["motility", "sporulation", "metabolism__aerobic", "metabolism__facultative", "carb_maltose"],
]
HIGHER_GROUPS = [
    ["metabolism__anaerobic", "cell_shape__bacillus", "cell_shape__coccus", "motility"],
    ["gram_stain", "carb_citrate", "carb_glucose", "carb_glycerol", "carb_lactose", "carb_sucrose"],
    ["sporulation", "metabolism__aerobic", "metabolism__facultative", "carb_maltose"],
]


def read_matrix(metric: str, variant: str) -> pd.DataFrame:
    return pd.read_csv(INPUT / f"interaction_{metric.lower()}_{variant}.csv", index_col=0)


def display_group(group: list[str] | tuple[str, ...]) -> str:
    return ", ".join(SHORT[x] for x in group)


def group_score(matrix: np.ndarray, indices: tuple[int, ...]) -> tuple[float, int, int, int, float, float]:
    """Directed score and signed accounting for one group; NaNs remain unobserved."""
    block = matrix[np.ix_(indices, indices)].copy()
    np.fill_diagonal(block, np.nan)
    known = np.isfinite(block)
    pos = block[(block > 0) & known]
    neg = block[(block < 0) & known]
    return (float(np.nansum(block)), int(known.sum()), int((pos > 0).sum()), int((neg < 0).sum()),
            float(pos.sum()), float(neg.sum()))


def solve_exact(matrix_df: pd.DataFrame, k: int, top_n: int = 10) -> list[tuple[float, tuple[int, ...]]]:
    """Top-N exact partitions using a canonical subset DP.

    Each recursive group must contain the lowest-index remaining task.  That
    removes group-order duplicates, so every legal unlabeled partition appears
    exactly once.  Retaining Top-N at each state is sufficient because a fixed
    first group adds the same constant to every completion.
    """
    matrix = matrix_df.to_numpy(float)
    n = len(matrix_df)
    all_mask = (1 << n) - 1
    cache: dict[int, float] = {}

    def bits(mask: int) -> tuple[int, ...]:
        return tuple(i for i in range(n) if mask & (1 << i))

    def score(mask: int) -> float:
        if mask not in cache:
            cache[mask] = group_score(matrix, bits(mask))[0]
        return cache[mask]

    @lru_cache(maxsize=None)
    def dp(remaining: int, groups_left: int) -> tuple[tuple[float, tuple[int, ...]], ...]:
        count = remaining.bit_count()
        if groups_left == 0:
            return ((0.0, ()),) if count == 0 else ()
        if count < MIN_SIZE * groups_left or count > MAX_SIZE * groups_left:
            return ()
        anchor = (remaining & -remaining).bit_length() - 1
        others = [i for i in range(anchor + 1, n) if remaining & (1 << i)]
        candidates: list[tuple[float, tuple[int, ...]]] = []
        for size in range(MIN_SIZE, MAX_SIZE + 1):
            if count - size < MIN_SIZE * (groups_left - 1) or count - size > MAX_SIZE * (groups_left - 1):
                continue
            for tail in itertools.combinations(others, size - 1):
                mask = (1 << anchor) | sum(1 << i for i in tail)
                for rest_score, rest_groups in dp(remaining ^ mask, groups_left - 1):
                    candidates.append((score(mask) + rest_score, (mask,) + rest_groups))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return tuple(candidates[:top_n])

    result = list(dp(all_mask, k))
    assert result, f"No legal {k}-group partition"
    for _, masks in result:
        assert len(masks) == k and sum(mask.bit_count() for mask in masks) == n
        assert all(MIN_SIZE <= mask.bit_count() <= MAX_SIZE for mask in masks)
        assert sum(masks) == all_mask and len(set(masks)) == k
    return result


def masks_to_groups(masks: tuple[int, ...], tasks: list[str]) -> list[list[str]]:
    return [[tasks[i] for i in range(len(tasks)) if mask & (1 << i)] for mask in masks]


def assignment(groups: list[list[str]], tasks: list[str]) -> list[int]:
    group_of = {task: group_id for group_id, group in enumerate(groups) for task in group}
    return [group_of[task] for task in tasks]


def partition_score(matrix: np.ndarray, groups: list[list[str]], tasks: list[str]) -> float:
    loc = {task: i for i, task in enumerate(tasks)}
    return sum(group_score(matrix, tuple(loc[x] for x in group))[0] for group in groups)


def group_details(matrix: np.ndarray, groups: list[list[str]], tasks: list[str]) -> pd.DataFrame:
    loc = {task: i for i, task in enumerate(tasks)}
    rows = []
    for group_id, group in enumerate(groups, start=1):
        idx = tuple(loc[x] for x in group)
        score, known, pos_n, neg_n, pos_sum, neg_sum = group_score(matrix, idx)
        outside = tuple(i for i in range(len(tasks)) if i not in idx)
        outbound = float(np.nansum(matrix[np.ix_(idx, outside)]))
        inbound = float(np.nansum(matrix[np.ix_(outside, idx)]))
        rows.append({"group": group_id, "size": len(group), "tasks": " | ".join(group),
                     "internal_score": score, "known_directed_edges": known,
                     "positive_edges": pos_n, "negative_edges": neg_n,
                     "positive_weight_sum": pos_sum, "negative_weight_sum": neg_sum,
                     "outbound_cross_score": outbound, "inbound_cross_score": inbound})
    return pd.DataFrame(rows)


def group_flow(matrix: np.ndarray, groups: list[list[str]], tasks: list[str]) -> pd.DataFrame:
    loc = {task: i for i, task in enumerate(tasks)}
    values = np.zeros((len(groups), len(groups)))
    for a, source_group in enumerate(groups):
        for b, target_group in enumerate(groups):
            values[a, b] = np.nansum(matrix[np.ix_([loc[x] for x in source_group], [loc[x] for x in target_group])])
    return pd.DataFrame(values, index=[f"G{i+1}" for i in range(len(groups))], columns=[f"G{i+1}" for i in range(len(groups))])


def cut_details(matrix: np.ndarray, groups: list[list[str]], tasks: list[str]) -> pd.DataFrame:
    """Directed signed cut accounting, including both directions between groups."""
    loc = {task: i for i, task in enumerate(tasks)}
    rows = []
    for a, source_group in enumerate(groups, start=1):
        for b, target_group in enumerate(groups, start=1):
            if a == b:
                continue
            block = matrix[np.ix_([loc[x] for x in source_group], [loc[x] for x in target_group])]
            known = np.isfinite(block)
            pos, neg = block[(block > 0) & known], block[(block < 0) & known]
            rows.append({"source_group": a, "target_group": b, "known_edges": int(known.sum()),
                         "cut_score": float(np.nansum(block)), "positive_edges": int((pos > 0).sum()),
                         "negative_edges": int((neg < 0).sum()), "positive_weight_sum": float(pos.sum()),
                         "negative_weight_sum": float(neg.sum())})
    return pd.DataFrame(rows)


def random_baseline(matrix: np.ndarray, groups: list[list[str]], tasks: list[str]) -> tuple[np.ndarray, float]:
    sizes = [len(group) for group in groups]
    scores = np.empty(N_RANDOM)
    task_array = np.array(tasks)
    for rep in range(N_RANDOM):
        shuffled = RNG.permutation(task_array)
        cursor, partition = 0, []
        for size in sizes:
            partition.append(shuffled[cursor:cursor + size].tolist())
            cursor += size
        scores[rep] = partition_score(matrix, partition, tasks)
    observed = partition_score(matrix, groups, tasks)
    percentile = float((1 + np.sum(scores <= observed)) / (N_RANDOM + 1))
    return scores, percentile


def swap_sensitivity(matrix: np.ndarray, groups: list[list[str]], tasks: list[str]) -> pd.DataFrame:
    base = partition_score(matrix, groups, tasks)
    rows = []
    for a, group_a in enumerate(groups):
        for task in group_a:
            candidates = []
            for b, group_b in enumerate(groups):
                if a == b:
                    continue
                for partner in group_b:
                    altered = [list(group) for group in groups]
                    altered[a][altered[a].index(task)] = partner
                    altered[b][altered[b].index(partner)] = task
                    candidates.append((base - partition_score(matrix, altered, tasks), partner, b + 1))
            loss, partner, target_group = min(candidates, key=lambda value: value[0])
            rows.append({"task": task, "current_group": a + 1, "best_swap_partner": partner,
                         "swap_to_group": target_group, "objective_loss_after_best_swap": loss})
    return pd.DataFrame(rows).sort_values("objective_loss_after_best_swap")


def draw_graph(matrix: np.ndarray, groups: list[list[str]], tasks: list[str], path: Path, title: str) -> None:
    """Readable signed graph: all internal edges with |I| >= 0.01, plus groups."""
    loc = {task: i for i, task in enumerate(tasks)}
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    graph = nx.DiGraph()
    for group_id, group in enumerate(groups):
        for task in group:
            graph.add_node(task, group=group_id)
    for group in groups:
        for source in group:
            for target in group:
                value = matrix[loc[source], loc[target]]
                if source != target and np.isfinite(value) and abs(value) >= 0.01:
                    graph.add_edge(source, target, weight=value)
    pos = nx.spring_layout(graph.to_undirected(), seed=42, k=1.35)
    fig, ax = plt.subplots(figsize=(13, 9))
    nx.draw_networkx_nodes(graph, pos, node_color=[colors[graph.nodes[x]["group"]] for x in graph.nodes],
                           node_size=1900, edgecolors="white", linewidths=1.6, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels={x: SHORT[x] for x in graph.nodes}, font_size=8, ax=ax)
    for sign, color in ((1, "#1b9e77"), (-1, "#d95f02")):
        edges = [(u, v) for u, v, d in graph.edges(data=True) if d["weight"] * sign > 0]
        widths = [1 + 24 * abs(graph[u][v]["weight"]) for u, v in edges]
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=color, width=widths,
                               alpha=.72, arrows=True, arrowstyle="-|>", arrowsize=14,
                               connectionstyle="arc3,rad=0.13", ax=ax)
    ax.set_title(title + "\ninternal edges only; |I| ≥ 0.01; green=positive, orange=negative")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def draw_random(scores: np.ndarray, observed: float, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(scores, bins=45, color="#9ecae1", edgecolor="white")
    ax.axvline(observed, color="#cb181d", linewidth=2.4, label=f"exact optimum = {observed:.4f}")
    ax.set(xlabel="LOTO signed within-group objective", ylabel="random legal partitions", title=title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def draw_flow(flow: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    limit = max(.02, float(np.abs(flow.to_numpy()).max()))
    sns.heatmap(flow, annot=True, fmt=".3f", cmap="vlag", center=0, vmin=-limit, vmax=limit,
                square=True, cbar_kws={"label": "sum directed LOTO interaction"}, ax=ax)
    ax.set(xlabel="Target group", ylabel="Source group", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def load_historical() -> tuple[pd.DataFrame, dict[str, list[list[str]]]]:
    rows, groupings = [], {}
    for name, expected in (("transfer", TRANSFER_GROUPS), ("higher_order", HIGHER_GROUPS)):
        summaries = sorted((HISTORICAL / name).glob("*/summary.json"))
        metrics = {}
        for summary_path in summaries:
            data = json.loads(summary_path.read_text())
            metrics.update({task: values["BAcc"] for task, values in data["test_per_task"].items()})
        groupings[name] = expected
        rows.append({"grouping": name, "n_groups": len(expected), "group_sizes": "-".join(map(str, map(len, expected))),
                     "actual_seed": 42, "actual_mean_bacc": float(np.mean(list(metrics.values()))),
                     "actual_n_tasks": len(metrics), "source_summary_count": len(summaries)})
    return pd.DataFrame(rows), groupings


def markdown_table(frame: pd.DataFrame, cols: list[str] | None = None, n: int | None = None) -> str:
    shown = frame if cols is None else frame[cols]
    if n is not None:
        shown = shown.head(n)
    def fmt(x: object) -> str:
        if isinstance(x, (float, np.floating)):
            return "" if np.isnan(x) else f"{x:.4f}"
        return str(x).replace("|", ",")
    rows = [[fmt(value) for value in row] for row in shown.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(map(str, shown.columns)) + " |",
                      "| " + " | ".join(["---"] * len(shown.columns)) + " |",
                      *["| " + " | ".join(row) + " |" for row in rows]])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    primary = read_matrix("bacc", "seed42")
    tasks = primary.index.tolist()
    assert len(tasks) == 14 and set(tasks) == set(SHORT), "Frozen 14-task panel changed"
    matrices = {(metric, variant): read_matrix(metric, variant).reindex(index=tasks, columns=tasks)
                for metric in ("bacc", "auroc") for variant in (*[f"seed{s}" for s in SEEDS], "mean")}

    solutions: dict[tuple[str, str, int], list[tuple[float, tuple[int, ...]]]] = {}
    partition_rows, detail_frames, cut_frames, random_rows, sensitivity_frames = [], [], [], [], []
    for (metric, variant), frame in matrices.items():
        matrix = frame.to_numpy(float)
        for k in (3, 4):
            top = solve_exact(frame, k)
            solutions[(metric, variant, k)] = top
            for rank, (score, masks) in enumerate(top, start=1):
                groups = masks_to_groups(masks, tasks)
                partition_rows.append({"metric": metric.upper(), "variant": variant, "k": k, "rank": rank,
                                       "objective": score, "group_sizes": "-".join(map(str, map(len, groups))),
                                       **{f"group_{i+1}": " | ".join(group) for i, group in enumerate(groups)}})
            best_score, best_masks = top[0]
            best_groups = masks_to_groups(best_masks, tasks)
            detail = group_details(matrix, best_groups, tasks)
            detail.insert(0, "variant", variant); detail.insert(0, "metric", metric.upper()); detail.insert(2, "k", k)
            detail_frames.append(detail)
            cut = cut_details(matrix, best_groups, tasks)
            cut.insert(0, "variant", variant); cut.insert(0, "metric", metric.upper()); cut.insert(2, "k", k)
            cut_frames.append(cut)
            sensitivity = swap_sensitivity(matrix, best_groups, tasks)
            sensitivity.insert(0, "variant", variant); sensitivity.insert(0, "metric", metric.upper()); sensitivity.insert(2, "k", k)
            sensitivity_frames.append(sensitivity)
            if variant == "mean":
                flow = group_flow(matrix, best_groups, tasks)
                flow.to_csv(OUT / f"{metric}_mean_k{k}_group_flow.csv")
                draw_graph(matrix, best_groups, tasks, PLOTS / f"{metric}_mean_k{k}_signed_graph.png",
                           f"{metric.upper()} exact K={k} partition")
                draw_flow(flow, PLOTS / f"{metric}_mean_k{k}_group_flow.png", f"{metric.upper()} K={k}: group-level flow")
                scores, percentile = random_baseline(matrix, best_groups, tasks)
                pd.DataFrame({"objective": scores}).to_csv(OUT / f"{metric}_mean_k{k}_random_matched_size_scores.csv", index=False)
                random_rows.append({"metric": metric.upper(), "variant": variant, "k": k, "n_random": N_RANDOM,
                                    "matched_group_sizes": "-".join(map(str, map(len, best_groups))), "optimum": best_score,
                                    "random_mean": scores.mean(), "random_sd": scores.std(ddof=1),
                                    "random_max": scores.max(), "percentile": percentile,
                                    "empirical_p_one_sided": (1 + np.sum(scores >= best_score)) / (N_RANDOM + 1)})
                draw_random(scores, best_score, PLOTS / f"{metric}_mean_k{k}_random_baseline.png",
                            f"{metric.upper()} K={k}: matched-size random baseline")

    partitions = pd.DataFrame(partition_rows)
    partitions.to_csv(OUT / "top10_exact_partitions.csv", index=False)
    details = pd.concat(detail_frames, ignore_index=True)
    details.to_csv(OUT / "optimal_group_internal_signed_summary.csv", index=False)
    pd.concat(cut_frames, ignore_index=True).to_csv(OUT / "optimal_group_cross_group_cut_summary.csv", index=False)
    sensitivity = pd.concat(sensitivity_frames, ignore_index=True)
    sensitivity.to_csv(OUT / "task_swap_sensitivity.csv", index=False)
    random_df = pd.DataFrame(random_rows)
    random_df.to_csv(OUT / "matched_size_random_baseline.csv", index=False)

    # BAcc seed and metric robustness.  Seed44 has two missing source rows; no values are imputed.
    stability_rows, cogroup_rows = [], []
    for k in (3, 4):
        labels = {}
        for variant in (*[f"seed{s}" for s in SEEDS], "mean"):
            groups = masks_to_groups(solutions[("bacc", variant, k)][0][1], tasks)
            labels[variant] = assignment(groups, tasks)
        for left, right in itertools.combinations(labels, 2):
            stability_rows.append({"analysis": "BAcc seed stability", "k": k, "left": left, "right": right,
                                   "ari": adjusted_rand_score(labels[left], labels[right]),
                                   "nmi": normalized_mutual_info_score(labels[left], labels[right])})
        for i, source in enumerate(tasks):
            for j, target in enumerate(tasks):
                if j <= i:
                    continue
                together = sum(label[i] == label[j] for label in labels.values())
                cogroup_rows.append({"k": k, "task_a": source, "task_b": target, "together_variants": together,
                                     "n_variants": len(labels), "co_group_frequency": together / len(labels)})
        bacc_mean_groups = masks_to_groups(solutions[("bacc", "mean", k)][0][1], tasks)
        auroc_mean_groups = masks_to_groups(solutions[("auroc", "mean", k)][0][1], tasks)
        bacc_labels, auroc_labels = assignment(bacc_mean_groups, tasks), assignment(auroc_mean_groups, tasks)
        stability_rows.append({"analysis": "BAcc vs AUROC mean", "k": k, "left": "bacc_mean", "right": "auroc_mean",
                               "ari": adjusted_rand_score(bacc_labels, auroc_labels),
                               "nmi": normalized_mutual_info_score(bacc_labels, auroc_labels)})
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(OUT / "partition_stability_ari_nmi.csv", index=False)
    cogroup = pd.DataFrame(cogroup_rows)
    cogroup.to_csv(OUT / "bacc_seed_cogroup_frequency.csv", index=False)

    # How much the membership changes among near-optimal multi-seed BAcc partitions.
    near_rows = []
    for k in (3, 4):
        top_groups = [masks_to_groups(masks, tasks) for _, masks in solutions[("bacc", "mean", k)]]
        for i, a in enumerate(tasks):
            for j, b in enumerate(tasks):
                if j <= i:
                    continue
                freq = np.mean([assignment(groups, tasks)[i] == assignment(groups, tasks)[j] for groups in top_groups])
                near_rows.append({"k": k, "task_a": a, "task_b": b, "top10_cogroup_frequency": freq})
    pd.DataFrame(near_rows).to_csv(OUT / "bacc_mean_top10_cogroup_frequency.csv", index=False)

    historical_actual, historical_groups = load_historical()
    historical_rows, historical_detail = [], []
    mean_bacc = matrices[("bacc", "mean")].to_numpy(float)
    exact_bacc_k3 = masks_to_groups(solutions[("bacc", "mean", 3)][0][1], tasks)
    for name, groups in historical_groups.items():
        score = partition_score(mean_bacc, groups, tasks)
        baseline = group_details(mean_bacc, groups, tasks)
        baseline.insert(0, "grouping", name)
        historical_detail.append(baseline)
        internal_slots = sum(len(group) * (len(group) - 1) for group in groups)
        historical_rows.append({"grouping": name, "loto_objective_bacc_mean": score,
                                "internal_directed_edge_slots": internal_slots,
                                "objective_per_internal_edge": score / internal_slots,
                                "ari_vs_loto_optimal_k3": adjusted_rand_score(assignment(groups, tasks), assignment(exact_bacc_k3, tasks)),
                                "nmi_vs_loto_optimal_k3": normalized_mutual_info_score(assignment(groups, tasks), assignment(exact_bacc_k3, tasks))})
    historical = historical_actual.merge(pd.DataFrame(historical_rows), on="grouping")
    historical.to_csv(OUT / "historical_grouping_comparison.csv", index=False)
    pd.concat(historical_detail, ignore_index=True).to_csv(OUT / "historical_grouping_signed_summary.csv", index=False)

    # Script-level checks: the optimum must beat all historical partitions and every random draw by construction.
    best_k3 = solutions[("bacc", "mean", 3)][0][0]
    assert best_k3 >= historical["loto_objective_bacc_mean"].max() - 1e-12
    assert all(row["optimum"] >= row["random_max"] - 1e-12 for row in random_rows)

    b3 = partitions.query("metric == 'BACC' and variant == 'mean' and k == 3 and rank == 1").iloc[0]
    b4 = partitions.query("metric == 'BACC' and variant == 'mean' and k == 4 and rank == 1").iloc[0]
    a3 = partitions.query("metric == 'AUROC' and variant == 'mean' and k == 3 and rank == 1").iloc[0]
    a4 = partitions.query("metric == 'AUROC' and variant == 'mean' and k == 4 and rank == 1").iloc[0]
    bacc_seed = stability.query("analysis == 'BAcc seed stability'")
    k3_seed_ari = bacc_seed.query("k == 3").ari
    k4_seed_ari = bacc_seed.query("k == 4").ari
    bacc_auroc_ari_k3 = stability.query("analysis == 'BAcc vs AUROC mean' and k == 3").iloc[0].ari
    bacc_auroc_ari_k4 = stability.query("analysis == 'BAcc vs AUROC mean' and k == 4").iloc[0].ari
    k3_second_gap = b3.objective - partitions.query("metric == 'BACC' and variant == 'mean' and k == 3 and rank == 2").iloc[0].objective
    k4_second_gap = b4.objective - partitions.query("metric == 'BACC' and variant == 'mean' and k == 4 and rank == 2").iloc[0].objective
    report = f"""# LOTO-guided global task partitioning

## Reproducibility

Run `python analyze_loto_guided_partitioning.py` from this directory with Python 3.12+ and NumPy, pandas, matplotlib, seaborn, scikit-learn, and networkx available. The script consumes only the frozen LOTO-analysis CSVs in the sibling `20261018_loto_transfer_interaction` directory and writes only this directory's `outputs/` artifacts. It has assertions for the 14-task panel, legal-cover constraints, historical-score dominance, and exact-optimum-vs-random checks.

## Scope and audit

This is a local CPU-only retrospective analysis. No model, original experiment artifact, scheduler state, or GPU job was changed. It uses the previously audited 14-task Full-HPS/LOTO result snapshot. The available LOTO interactions are **test-set** BAcc/AUROC differences, not validation per-task interactions; therefore the partition search is exploratory and reuses test information. No validation interaction matrix was found in the saved summaries.

`I[i,j] = BAcc_full[j] - BAcc_LOTO_without_i[j]` is directed: a positive value means source `i` helped target `j` in the Full-HPS context. The 42/43/44 comparisons are seed-matched. The multi-seed mean uses the observed matched runs only: `carb_maltose` and `carb_sucrose` have two runs; seed 44 has no outgoing rows for those two sources. Their missing values were **not** filled; the seed-44 objective therefore covers fewer directed edges and is not score-comparable to complete seeds.

Taskonomy Section 3.4 motivated the use of an exact global graph optimization, but its BIP selects a transfer subgraph under a supervision budget. This analysis instead solves the user's different, fixed-partition objective exactly; it does not claim to reproduce the Taskonomy taxonomy formulation.

## Objective and exact solver

For every legal partition $\\mathcal{{G}}$ with K groups of 3–6 tasks, the optimized proxy is

$$F(\\mathcal{{G}})=\\sum_{{G\\in\\mathcal{{G}}}}\\sum_{{i\\ne j; i,j\\in G}} I_{{i\\to j}}.$$

It follows from the stated first-order additive approximation: retaining a source in a target's group retains its observed Full-HPS marginal contribution. Both directions are counted for a co-grouped pair and negative edges are retained. A canonical subset dynamic program enumerates every feasible unlabeled partition once, retaining the top 10 completions per state; group-size and complete-cover assertions are built into the script. This is exact for $F$, not a guarantee for actual simultaneous multi-task removal/training.

## Primary multi-seed BAcc optima

### K=3

Objective **{b3.objective:.4f}**, group sizes **{b3.group_sizes}**.

1. {display_group(b3.group_1.split(' | '))}
2. {display_group(b3.group_2.split(' | '))}
3. {display_group(b3.group_3.split(' | '))}

### K=4

Objective **{b4.objective:.4f}**, group sizes **{b4.group_sizes}**.

1. {display_group(b4.group_1.split(' | '))}
2. {display_group(b4.group_2.split(' | '))}
3. {display_group(b4.group_3.split(' | '))}
4. {display_group(b4.group_4.split(' | '))}

The full top-10 exact candidates are in `top10_exact_partitions.csv`; group signed accounting and single-swap sensitivity are in the accompanying CSVs. Figures show the readable internal edges ($|I|\\ge0.01$) rather than pretending that an all-182-edge drawing is interpretable.

## Robustness checks

AUROC produces the following mean-matrix optima: K=3, **{a3.objective:.4f}** ({a3.group_sizes}); K=4, **{a4.objective:.4f}** ({a4.group_sizes}). This is not strong membership replication: the complete BAcc-seed pairwise ARIs span **{k3_seed_ari.min():.3f} to {k3_seed_ari.max():.3f}** for K=3 and **{k4_seed_ari.min():.3f} to {k4_seed_ari.max():.3f}** for K=4; BAcc-vs-AUROC mean ARI is **{bacc_auroc_ari_k3:.3f}** (K=3) and **{bacc_auroc_ari_k4:.3f}** (K=4). The runner-up BAcc objective is only {k3_second_gap:.6f} below the K=3 optimum and {k4_second_gap:.6f} below K=4, so several allocations are effectively near-tied. The full ARI/NMI table is `partition_stability_ari_nmi.csv`, and seed/top-10 co-group frequencies are supplied separately. Seed-44 is partial as described above.

Each primary BAcc/AUROC solution was compared with {N_RANDOM:,} random legal partitions having exactly the same group sizes. The exact optimum exceeded every draw in this finite null sample; the empirical one-sided p-value is bounded by $1/({N_RANDOM}+1)$, but this is a randomized-combinatorial reference, not an independent biological significance test.

## Historical grouped-MTL comparison

{markdown_table(historical, n=2)}

The completed historical runs are only seed 42 and were selected/evaluated with their own early-stopping runs, whereas this LOTO objective is a Full-HPS marginal proxy. With only two historical groupings this comparison can test ordering but cannot validate the proxy generally. In particular, the report must not call a match (or mismatch) causal evidence without prospective grouped training.

## Interpretation and recommendation

**Observation:** The exact optimizer preferentially retains directed positive LOTO edges, including several hub-like outgoing effects; it also actively avoids observed negative pairs. Same-label-derived groups—Bacillus/Coccus and Aerobic/Anaerobic/Facultative—are flagged as dependent phenotype encodings, not independent biological discoveries.

**Statistical support:** random matched-size partitions establish that the reported $F$ optima are non-random under this particular edge matrix. Seed/AUROC membership agreement, edge coverage, top-10 co-group frequencies, and swap losses determine whether any particular allocation is stable; they are reported rather than averaged away.

**Recommendation:** the current data do **not** support a stable whole-panel taxonomy, so there is no high-confidence training recommendation. If one prospective experiment must be pre-registered, use the stated **K=3 BAcc solution** as the primary *exploratory* candidate, because it is the exact mean-BAcc optimum and the proxy correctly ranks the two already-run historical schemes. Keep the stated **K=4 BAcc solution** only as a capacity-control alternate. The persistent local core is Motility–Gram stain–Bacillus–Coccus–Anaerobic, but it is partly composed of related label encodings and is not evidence of a new biological module. The risk is fundamental: every edge was measured after removing one task from Full HPS, whereas a group removes several sources jointly; interaction non-additivity and test reuse can reverse the ranking. A future grouped-MTL trial—not more post-hoc graph optimization—is required to validate the proxy.
"""
    (OUT / "LOTO_guided_partitioning_report.md").write_text(report)


if __name__ == "__main__":
    main()
