#!/usr/bin/env python3
"""Build validation-only calibrated LOTO graphs and reuse the locked exact solver.

The input is the completed, evaluation-only export from PBS job 38197.  This
script neither opens a model checkpoint nor reads any test metric when building
an interaction graph.  It imports the original signed directed subset-DP solver
unchanged, preserving the K=3/K=4 and 3--6-task-group constraints.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "inputs" / "checkpoint_evaluation"
OUT = ROOT / "outputs"
PLOTS = OUT / "plots"
ORIGINAL = ROOT.parent / "20261018_loto_guided_partitioning" / "analyze_loto_guided_partitioning.py"


def load_solver():
    spec = importlib.util.spec_from_file_location("locked_loto_partition_solver", ORIGINAL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOLVER = load_solver()


def heatmap(frame: pd.DataFrame, path: Path, title: str, label: str) -> None:
    limit = max(0.01, float(np.nanmax(np.abs(frame.to_numpy(float)))))
    fig, ax = plt.subplots(figsize=(10.5, 8.7))
    sns.heatmap(frame, cmap="vlag", center=0, vmin=-limit, vmax=limit, square=True,
                xticklabels=[SOLVER.SHORT[x] for x in frame.columns],
                yticklabels=[SOLVER.SHORT[x] for x in frame.index],
                cbar_kws={"label": label}, ax=ax)
    ax.set(xlabel="Target task", ylabel="Removed source task", title=title)
    ax.tick_params(axis="x", rotation=55)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def matrix_from_metrics(metrics: pd.DataFrame, value_column: str, tasks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return I[source,target] = Full(target) - LOTO-without-source(target)."""
    full = metrics.loc[metrics.model_id.eq("full_hps_mask_seed42")].set_index("task")
    if set(full.index) != set(tasks) or not full.index.is_unique:
        raise ValueError("Full-HPS task panel is incomplete or duplicated")
    matrix = pd.DataFrame(np.nan, index=tasks, columns=tasks, dtype=float)
    edges = []
    for source in tasks:
        loto = metrics.loc[metrics.removed_task.eq(source)].set_index("task")
        expected = set(tasks) - {source}
        if set(loto.index) != expected or not loto.index.is_unique:
            raise ValueError(f"LOTO without {source} has an inconsistent target panel")
        for target in tasks:
            if source == target:
                continue
            full_value, loto_value = float(full.loc[target, value_column]), float(loto.loc[target, value_column])
            value = full_value - loto_value
            matrix.loc[source, target] = value
            edges.append({"source": source, "target": target, "full_validation_metric": full_value,
                          "loto_without_source_validation_metric": loto_value, "interaction": value})
    # The frame is initialized with NaNs, so its diagonal is already undefined.
    return matrix, pd.DataFrame(edges)


def assignment(groups: list[list[str]], tasks: list[str]) -> list[int]:
    group_of = {task: index for index, group in enumerate(groups) for task in group}
    return [group_of[task] for task in tasks]


def top_partitions(name: str, frame: pd.DataFrame, tasks: list[str]) -> tuple[pd.DataFrame, list[dict], dict[int, list[list[str]]]]:
    rows, summaries, best = [], [], {}
    matrix = frame.to_numpy(float)
    for k in (3, 4):
        candidates = SOLVER.solve_exact(frame, k, top_n=10)
        for rank, (objective, masks) in enumerate(candidates, start=1):
            groups = SOLVER.masks_to_groups(masks, tasks)
            rows.append({"graph": name, "k": k, "rank": rank, "objective": objective,
                         "group_sizes": "-".join(str(len(group)) for group in groups),
                         **{f"group_{i + 1}": " | ".join(group) for i, group in enumerate(groups)}})
        groups = SOLVER.masks_to_groups(candidates[0][1], tasks)
        best[k] = groups
        summary = SOLVER.group_details(matrix, groups, tasks)
        summary.insert(0, "k", k)
        summary.insert(0, "graph", name)
        summaries.append(summary)
        flow = SOLVER.group_flow(matrix, groups, tasks)
        flow.to_csv(OUT / f"{name}_k{k}_group_flow.csv")
        SOLVER.draw_graph(matrix, groups, tasks, PLOTS / f"{name}_k{k}_signed_graph.png",
                          f"{name}: exact K={k} partition")
        SOLVER.draw_flow(flow, PLOTS / f"{name}_k{k}_group_flow.png", f"{name}: group-level flow")
    return pd.DataFrame(rows), summaries, best


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(INPUT / "per_model_task_metrics.csv")
    tasks = metrics.loc[metrics.model_id.eq("full_hps_mask_seed42"), "task"].tolist()
    if len(tasks) != 14 or len(set(tasks)) != 14 or set(tasks) != set(SOLVER.SHORT):
        raise ValueError("The frozen 14-task panel is missing or has changed")
    expected_rows = 14 + 14 * 13
    if len(metrics) != expected_rows:
        raise ValueError(f"Expected {expected_rows} model-task rows, found {len(metrics)}")

    specs = {
        "validation_fixed05_bacc": "val_fixed05_BAcc",
        "validation_calibrated_bacc": "val_tuned_BAcc",
        "validation_auprc": "val_fixed05_AUPRC",
        "validation_auroc": "val_fixed05_AUROC",
    }
    matrices, edges = {}, []
    audits = []
    for name, column in specs.items():
        matrix, edge = matrix_from_metrics(metrics, column, tasks)
        assert np.isnan(np.diag(matrix.to_numpy(float))).all()
        assert int(np.isfinite(matrix.to_numpy(float)).sum()) == 14 * 13
        matrix.to_csv(OUT / f"interaction_{name}_seed42.csv")
        edge.insert(0, "graph", name)
        matrices[name] = matrix
        edges.append(edge)
        audits.append({"graph": name, "metric_column": column, "metric_split": "validation",
                       "n_tasks": len(tasks), "observed_directed_edges": int(np.isfinite(matrix.to_numpy(float)).sum()),
                       "expected_directed_edges": len(tasks) * (len(tasks) - 1),
                       "diagonal_all_nan": bool(np.isnan(np.diag(matrix.to_numpy(float))).all()),
                       "uses_test_to_select_threshold_or_graph": False})
        heatmap(matrix, PLOTS / f"{name}_interaction_heatmap.png", f"{name}: validation LOTO interaction", "Full − LOTO")
    edge_table = pd.concat(edges, ignore_index=True)
    edge_table.to_csv(OUT / "validation_interaction_edges.csv", index=False)
    pd.DataFrame(audits).to_csv(OUT / "validation_interaction_graph_audit.csv", index=False)

    # Quantify threshold calibration before optimizing any partition.
    correction = metrics.loc[:, ["model_id", "removed_task", "task", "threshold_validation_bacc",
                                  "val_fixed05_BAcc", "val_tuned_BAcc"]].copy()
    correction["bacc_change_tuned_minus_fixed05"] = correction.val_tuned_BAcc - correction.val_fixed05_BAcc
    correction.to_csv(OUT / "validation_bacc_threshold_correction_per_model_task.csv", index=False)
    correction.groupby("task", sort=False).agg(
        n=("task", "size"), mean_fixed05_bacc=("val_fixed05_BAcc", "mean"),
        mean_calibrated_bacc=("val_tuned_BAcc", "mean"),
        mean_bacc_change=("bacc_change_tuned_minus_fixed05", "mean"),
        max_bacc_change=("bacc_change_tuned_minus_fixed05", "max"),
        median_threshold=("threshold_validation_bacc", "median"),
    ).reset_index().to_csv(OUT / "validation_bacc_threshold_correction_by_task.csv", index=False)
    interaction_delta = matrices["validation_calibrated_bacc"] - matrices["validation_fixed05_bacc"]
    interaction_delta.to_csv(OUT / "interaction_delta_calibrated_minus_fixed05_bacc_seed42.csv")
    delta_edges = interaction_delta.stack().dropna().rename("interaction_change_calibrated_minus_fixed05").reset_index()
    delta_edges.columns = ["source", "target", "interaction_change_calibrated_minus_fixed05"]
    delta_edges["fixed05_interaction"] = [matrices["validation_fixed05_bacc"].loc[s, t] for s, t in zip(delta_edges.source, delta_edges.target)]
    delta_edges["calibrated_interaction"] = [matrices["validation_calibrated_bacc"].loc[s, t] for s, t in zip(delta_edges.source, delta_edges.target)]
    delta_edges["sign_changed"] = np.sign(delta_edges.fixed05_interaction) != np.sign(delta_edges.calibrated_interaction)
    delta_edges.sort_values("interaction_change_calibrated_minus_fixed05", key=np.abs, ascending=False).to_csv(
        OUT / "interaction_change_calibrated_minus_fixed05_ranked.csv", index=False)
    heatmap(interaction_delta, PLOTS / "calibrated_minus_fixed05_bacc_interaction_change.png",
            "Validation BAcc interaction change: calibrated − fixed 0.5", "interaction change")

    all_partitions, all_summaries, best = [], [], {}
    for name, matrix in matrices.items():
        partitions, summaries, groups = top_partitions(name, matrix, tasks)
        all_partitions.append(partitions)
        all_summaries.extend(summaries)
        best[name] = groups
    partitions = pd.concat(all_partitions, ignore_index=True)
    partitions.to_csv(OUT / "top10_exact_partitions.csv", index=False)
    pd.concat(all_summaries, ignore_index=True).to_csv(OUT / "optimal_group_internal_signed_summary.csv", index=False)

    # Directly compare the two validation BAcc operating-point choices.
    comparisons = []
    for k in (3, 4):
        fixed_groups, tuned_groups = best["validation_fixed05_bacc"][k], best["validation_calibrated_bacc"][k]
        fixed_obj = SOLVER.partition_score(matrices["validation_fixed05_bacc"].to_numpy(float), fixed_groups, tasks)
        tuned_obj = SOLVER.partition_score(matrices["validation_calibrated_bacc"].to_numpy(float), tuned_groups, tasks)
        comparisons.append({
            "k": k,
            "fixed05_optimum_objective_on_fixed_graph": fixed_obj,
            "calibrated_optimum_objective_on_calibrated_graph": tuned_obj,
            "fixed_partition_objective_on_calibrated_graph": SOLVER.partition_score(matrices["validation_calibrated_bacc"].to_numpy(float), fixed_groups, tasks),
            "calibrated_partition_objective_on_fixed_graph": SOLVER.partition_score(matrices["validation_fixed05_bacc"].to_numpy(float), tuned_groups, tasks),
            "ari_fixed_vs_calibrated": adjusted_rand_score(assignment(fixed_groups, tasks), assignment(tuned_groups, tasks)),
            "nmi_fixed_vs_calibrated": normalized_mutual_info_score(assignment(fixed_groups, tasks), assignment(tuned_groups, tasks)),
            **{f"fixed_group_{i + 1}": " | ".join(group) for i, group in enumerate(fixed_groups)},
            **{f"calibrated_group_{i + 1}": " | ".join(group) for i, group in enumerate(tuned_groups)},
        })
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(OUT / "fixed05_vs_calibrated_bacc_partition_comparison.csv", index=False)

    full = correction.loc[correction.model_id.eq("full_hps_mask_seed42")]
    primary = partitions.query("k == 3 and rank == 1").copy()
    primary.to_csv(OUT / "k3_primary_graph_candidates.csv", index=False)
    def group_lines(graph: str) -> list[str]:
        row = primary.loc[primary.graph.eq(graph)].iloc[0]
        return [f"{i}. {SOLVER.display_group(str(row[f'group_{i}']).split(' | '))}" for i in range(1, 4)]
    report = [
        "# Validation-calibrated masked LOTO graph optimization",
        "",
        "## Locked protocol",
        "",
        "PBS job 38197 performed checkpoint evaluation only for the masked Full-HPS seed42 model and 14 masked LOTO seed42 models. It did not train, modify a checkpoint, or use test labels to choose a threshold.",
        "For every model-task pair, the threshold maximizing validation BAcc was selected from validation probabilities, with deterministic ties closest to 0.5. Every graph in this directory uses validation metrics only.",
        "",
        "The interaction is `I[source, target] = validation_metric(Full-HPS, target) − validation_metric(LOTO without source, target)`. Rows are removed sources and columns are evaluated targets. Positive edges mean the source helped the target under the Full-HPS context.",
        "",
        "The exact solver is imported unchanged from the original LOTO partitioning analysis: signed directed within-group sum, all 14 tasks assigned once, K=3/K=4, group size 3–6, no normalization, and negative edges retained.",
        "",
        "## Threshold-correction magnitude",
        "",
        f"Across all 196 model-task validation evaluations, calibrated-minus-fixed-0.5 BAcc has mean {correction.bacc_change_tuned_minus_fixed05.mean():.6f}, median {correction.bacc_change_tuned_minus_fixed05.median():.6f}, and maximum {correction.bacc_change_tuned_minus_fixed05.max():.6f}.",
        f"For Full-HPS alone, the mean gain is {full.bacc_change_tuned_minus_fixed05.mean():.6f}; its largest task-level gain is {full.bacc_change_tuned_minus_fixed05.max():.6f}.",
        f"Among 182 directed edges, {int(delta_edges.sign_changed.sum())} change sign when replacing fixed-0.5 validation BAcc with validation-calibrated BAcc.",
        "",
        "## Deliverables",
        "",
        "- `interaction_validation_calibrated_bacc_seed42.csv`: primary calibrated-BAcc graph.",
        "- `interaction_validation_auprc_seed42.csv` and `interaction_validation_auroc_seed42.csv`: threshold-free validation robustness graphs.",
        "- `top10_exact_partitions.csv`: K=3/K=4 exact top-10 partitions for all four graph variants, including fixed-0.5 BAcc as the pre-correction comparison.",
        "- `fixed05_vs_calibrated_bacc_partition_comparison.csv`: explicit change in BAcc partition structure and cross-scoring.",
        "",
        "## Exact K=3 candidates",
        "",
        f"**Validation fixed-0.5 BAcc** (objective {primary.loc[primary.graph.eq('validation_fixed05_bacc'), 'objective'].iloc[0]:.6f}):",
        *group_lines("validation_fixed05_bacc"),
        "",
        f"**Validation calibrated-BAcc** (primary graph; objective {primary.loc[primary.graph.eq('validation_calibrated_bacc'), 'objective'].iloc[0]:.6f}):",
        *group_lines("validation_calibrated_bacc"),
        "",
        f"**Validation AUPRC** (objective {primary.loc[primary.graph.eq('validation_auprc'), 'objective'].iloc[0]:.6f}):",
        *group_lines("validation_auprc"),
        "",
        f"**Validation AUROC** (objective {primary.loc[primary.graph.eq('validation_auroc'), 'objective'].iloc[0]:.6f}):",
        *group_lines("validation_auroc"),
        "",
        "## Limitation",
        "",
        "This is one seed and uses the same validation data both to select each model-task threshold and to compute the interaction graph. It is appropriate for exploratory graph selection, but a selected partition still requires independent grouped-MTL confirmation. Test metrics are excluded from this optimization.",
    ]
    (OUT / "validation_calibrated_loto_partitioning_report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
