import wandb
import pandas as pd
import matplotlib.pyplot as plt

api = wandb.Api()
project_path = "efficient-attention-nju/shakespeare"
filters = {
    "$exists": [
        {"model/model_name": "conceptual_deltanet"},
        {"model/model_name": "omd_deltanet"},
        {"model/model_name": "deltanet"},
    ]
}
runs = api.runs(project_path, filters=filters)
runs = api.runs(project_path)
data_frames = []
for run in runs:
    summary = run.summary
    config = run.config
    model_name = config.get("model/model_name", "unknown_model")
    eta = config.get("model/eta", None)
    val_loss = summary.get("val/loss", None)
    if eta is not None and val_loss is not None:
        data_frames.append(pd.DataFrame({
            "model_name": [model_name],
            "model/eta": [eta],
            "val/loss": [val_loss]
        }))
df: pd.DataFrame = pd.concat(data_frames, ignore_index=True)
df.sort_values(by=["model_name", "model/eta"], inplace=True)
print(df)
df.to_csv("val_loss_vs_eta.csv", index=False)
plt.figure(figsize=(10, 6))
for model_name, group in df.groupby("model_name"):
    plt.plot(group["model/eta"], group["val/loss"], marker='o', label=model_name)
    plt.xscale("log")
plt.xlabel("Learning Rate (eta)")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs Learning Rate for Different Models")
plt.legend()
plt.grid(True)
plt.savefig("val_loss_vs_eta.png")