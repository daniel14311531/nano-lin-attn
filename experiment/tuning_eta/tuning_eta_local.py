import os
import pandas as pd
import matplotlib.pyplot as plt

res_dir = '../../out-shakespeare'

files = os.listdir(res_dir)
tmp = []
for file in files:
    if file.startswith('result'):
        tmp.append(file)
files = tmp

data_frames = []
for file in files:
    path = os.path.join(res_dir, file)
    with open(path, 'r') as f:
        content = f.read()
    model_name, eta, val_loss = content.strip().split()
    model_name = model_name
    eta = float(eta)
    val_loss = float(val_loss)
    data_frames.append(pd.DataFrame({
        "model_name": [model_name],
        "model/eta": [eta],
        "val/loss": [val_loss]
    }))
df: pd.DataFrame = pd.concat(data_frames, ignore_index=True)
df.sort_values(by=["model_name", "model/eta"], inplace=True)
print(df)
df.to_csv("val_loss_vs_eta_local.csv", index=False)
plt.figure(figsize=(10, 6))
for model_name, group in df.groupby("model_name"):
    plt.plot(group["model/eta"], group["val/loss"], marker='o', label=model_name)
    plt.xscale("log")
plt.xlabel("Learning Rate (eta)")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs Learning Rate for Different Models")
plt.legend()
plt.grid(True)
plt.savefig("val_loss_vs_eta_local.png")