#!/usr/bin/env python3
"""Parse lm-evaluation-harness results and generate a LaTeX table + JSON summary."""

import json
import os
import sys
import glob


def main():
    if len(sys.argv) < 2:
        print("Usage: python _parse_lm_eval_results.py <results_dir>", file=sys.stderr)
        sys.exit(1)

    results_dir = sys.argv[1]

    # lm-eval saves results in a subdirectory; find the most recent results JSON
    result_files = glob.glob(os.path.join(results_dir, "**", "results.json"), recursive=True)
    if not result_files:
        print(f"Warning: No results.json found in {results_dir}", file=sys.stderr)
        sys.exit(0)

    result_file = max(result_files, key=os.path.getmtime)
    print(f"Reading results from: {result_file}")

    with open(result_file, "r") as f:
        data = json.load(f)

    results = data.get("results", {})

    # Benchmark -> (display name, metric key, num_fewshot)
    BENCHMARKS = {
        "mmlu": ("MMLU", "acc,none", "5-shot"),
        "hellaswag": ("HellaSwag", "acc_norm,none", "10-shot"),
        "arc_challenge": ("ARC-C", "acc_norm,none", "25-shot"),
        "winogrande": ("WinoGrande", "acc,none", "5-shot"),
        "gsm8k": ("GSM8K", "exact_match,strict-match", "5-shot"),
        "truthfulqa_mc2": ("TruthfulQA", "acc,none", "0-shot"),
    }

    print()
    print("=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    scores = {}
    for task_key, (display_name, metric_key, fewshot) in BENCHMARKS.items():
        task_result = results.get(task_key, {})
        if not task_result:
            for key in results:
                if task_key in key:
                    task_result = results[key]
                    break

        score = task_result.get(metric_key)
        if score is None:
            for k, v in task_result.items():
                if "acc" in k and isinstance(v, (int, float)):
                    score = v
                    break

        if score is not None:
            scores[task_key] = score * 100
            print(f"  {display_name:12s} ({fewshot:>7s}): {scores[task_key]:6.2f}%")
        else:
            scores[task_key] = None
            print(f"  {display_name:12s} ({fewshot:>7s}):    N/A")

    valid_scores = [s for s in scores.values() if s is not None]
    avg = sum(valid_scores) / len(valid_scores) if valid_scores else 0
    print(f"  {'Average':12s}         : {avg:6.2f}%")
    print("=" * 70)

    # Build LaTeX table
    header_names = []
    score_cells = []
    for task_key, (display_name, _, _) in BENCHMARKS.items():
        header_names.append(display_name)
        s = scores.get(task_key)
        score_cells.append(f"{s:.1f}" if s is not None else "---")
    header_names.append("Avg.")
    score_cells.append(f"{avg:.1f}")

    tex_header = " & ".join(header_names)
    tex_scores = " & ".join(score_cells)
    ncols = len(header_names)

    tex_lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{l" + "c" * ncols + "}",
        r"\toprule",
        f"Model & {tex_header} \\\\",
        r"\midrule",
        f"Ours & {tex_scores} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Accuracy benchmarks on standard evaluation suites.}",
        r"\label{tab:accuracy}",
        r"\end{table}",
    ]

    print()
    print("LaTeX table:")
    print()
    for line in tex_lines:
        print(line)

    tex_path = os.path.join(results_dir, "accuracy_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"\nLaTeX table saved to: {tex_path}")

    summary = {
        "scores": {k: v for k, v in scores.items()},
        "average": avg,
        "source_file": result_file,
    }
    summary_path = os.path.join(results_dir, "accuracy_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
