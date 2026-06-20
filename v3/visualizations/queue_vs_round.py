import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = "../outputs/digital_twin"

files = {
    "Baseline": "global_metrics_baseline_live.csv",
    "No Fallback": "global_metrics_outage_no_fallback.csv",
    "Fixed Replay": "global_metrics_outage_replay_fixed.csv",
    "KDE Replay": "global_metrics_outage_replay_kde.csv",
    "WGAN Replay": "global_metrics_outage_replay_wgan.csv",
}

all_data = []

for label, fname in files.items():
    df = pd.read_csv(os.path.join(BASE_DIR, fname))
    df["scenario"] = label
    all_data.append(df[["round", "queue_fill_seconds_max", "scenario"]])

df_all = pd.concat(all_data, ignore_index=True)

heatmap = df_all.pivot_table(
    index="scenario",
    columns="round",
    values="queue_fill_seconds_max",
    aggfunc="mean"
)

plt.figure(figsize=(14, 4))
plt.imshow(heatmap, aspect="auto", interpolation="nearest")

plt.colorbar(label="Max Queue Fill Time (s)")

plt.yticks(range(len(heatmap.index)), heatmap.index)
plt.xticks(range(0, len(heatmap.columns), 10),
           heatmap.columns[::10],
           rotation=90)

plt.title("Queue Fill Time Across Scenarios (Heatmap)")
plt.xlabel("Round")
plt.ylabel("Scenario")

plt.tight_layout()
plt.savefig("figure_queue_fill_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()