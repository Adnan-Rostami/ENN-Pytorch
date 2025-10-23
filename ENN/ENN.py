 
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
torch.cuda.empty_cache()
from sklearn.metrics import accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== تنظیم پوشه ثبت خروجی ====
base_dir = "results_runs"
os.makedirs(base_dir, exist_ok=True)
run_id = len(os.listdir(base_dir)) + 1
run_folder = os.path.join(base_dir, f"run_{run_id:03d}")
os.makedirs(run_folder, exist_ok=True)
CSV_PATH = os.path.join(run_folder, f"run_{run_id:03d}.csv")

# ==== بارگذاری مستقیم داده‌های train / test ====
train_data = np.loadtxt('data.csv', delimiter=',', skiprows=1)    
test_data = np.loadtxt('data.csv', delimiter=',', skiprows=1)
 

print("Train unique labels:", np.unique(train_data[:, -1]))
print("Test  unique labels:", np.unique(test_data[:, -1]))

# ==== جداکردن X و Y ====
X_train = torch.tensor(train_data[:, :-1], dtype=torch.float32).to(device)
y_train = torch.tensor(train_data[:, -1],  dtype=torch.float32).unsqueeze(1).to(device)

X_test  = torch.tensor(test_data[:, :-1], dtype=torch.float32).to(device)
y_test  = torch.tensor(test_data[:, -1],  dtype=torch.float32).unsqueeze(1).to(device)

print("Train shape:", X_train.shape, y_train.shape)
print("Test  shape:", X_test.shape, y_test.shape)

# ==== مدل ENN ====

# # ==== مدل ENN ====
class ENN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            # nn.Sigmoid()
        )
    def forward(self, x):

        return self.net(x)

model = ENN(X_train.shape[1]).to(device)

μ = 0.2
β = 0.6
# loss.backward()
# optimizer = optim.SGD(model.parameters(), lr=μ)
# optimizer.step()

epochs = 20
e1 = float('inf')

records = []  # برای ذخیره نتایج هر epoch

start_time = time.time()
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([227450 / 394], device=device))
optimizer = optim.SGD(model.parameters(), lr=μ)
for epoch in range(epochs):
    model.train()
    
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    e2 = loss.item()

    # Adaptive μ
    if e2 >= e1:
        μ *= β
    else:
        μ /= β
    μ = float(np.clip(μ, 1e-5, 0.1))
    for g in optimizer.param_groups:
        g['lr'] = μ

    loss.backward()
    optimizer.step()
    e1 = e2

    # ارزیابی روی داده تست
model.eval()
with torch.no_grad():
    outputs_test  =  model(X_test) 
    probs = torch.sigmoid(outputs_test).detach().cpu().numpy() 
    preds = (probs >= 0.5).astype(int) 
    y_true = y_test.cpu().numpy()
    i = 1 
    # if(i == 1):
    
    #     break
    acc = accuracy_score(y_true, preds)
    # rmse = mean_squared_error(y_test.cpu(), preds.cpu(), squared=False)
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    cm = confusion_matrix(y_true, preds)
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
        
records.append({
    "epoch": epoch + 1,
    "loss": e2,
    "mu": μ,
    "accuracy": acc,
    "rmse": rmse ,
    # 'roc_auc':roc_auc,
    'precision':precision,
    'recall':recall,
    'f1':f1,
    'roc_auc':roc_auc,
    'pr_auc':pr_auc
    # "cm": cm
})

print(f"Epoch {epoch+1:03d} | Loss={e2:.6f} | μ={μ:.6f}")
print(cm)
end_time = time.time()
# records.append(end_time)
final_cm = cm

unique, counts = np.unique(y_train.cpu().numpy(), return_counts=True)
print(dict(zip(unique, counts)))

unique_t, counts_t = np.unique(y_test.cpu().numpy(), return_counts=True)

# ==== ثبت نتایج به CSV ====
df = pd.DataFrame(records)
df.to_csv(CSV_PATH, index=False)
# df2 = pd.DataFrame(final_cm,
#                    columns=[ 'Pred_Neg', 'Pred_Pos'],
#                    index=[ 'True_Neg', 'True_Pos'])
cm_expanded = np.vstack([
    ['', 'Pred_Neg', 'Pred_Pos'],
    ['True_Neg', final_cm[0,0], final_cm[0,1]],
    ['True_Pos', final_cm[1,0], final_cm[1,1]],
    ['', '', ''],                     # ردیف فاصله
    ['Run_Time_sec', end_time, '']
])

# ساخت DataFrame بدون ستون و ایندکس اضافه
df2 = pd.DataFrame(cm_expanded)
# df2.loc[''] = ['', '']                       # ردیف خالی برای فاصله
# df2.loc['Run_Time_sec'] = [round(end_time - start_time, 4), 'Second to run']
CSV_PATH2 = os.path.join(run_folder, f"confusion_run_{run_id:03d}.csv")
df2.to_csv(CSV_PATH2, index=False)
print(f"Results saved at: {CSV_PATH2}")

# ==== رسم نمودار دقت ====
plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["accuracy"], marker='o', label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Run {run_id} - Accuracy per Epoch")
plt.grid(True)
plt.legend()
IMG_PATH = os.path.join(run_folder, f"accuracy_run_{run_id:03d}.png")
plt.savefig(IMG_PATH, dpi=200)
plt.close()
print(f"Accuracy plot saved at: {IMG_PATH}")

#pr_auc
plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["pr_auc"], marker='o', label="pr_auc")
plt.xlabel("Epoch")
plt.ylabel("pr_auc")
plt.title(f"Run {run_id} - pr_auc per Epoch")
plt.grid(True)
plt.legend()
IMG_PATH = os.path.join(run_folder, f"pr_auc_run_{run_id:03d}.png")
plt.savefig(IMG_PATH, dpi=200)
plt.close()
print(f"pr_auc plot saved at: {IMG_PATH}")

#roc_auc
plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["roc_auc"], marker='o', label="roc_auc")
plt.xlabel("Epoch")
plt.ylabel("roc_auc")
plt.title(f"Run {run_id} - roc_auc per Epoch")
plt.grid(True)
plt.legend()
IMG_PATH = os.path.join(run_folder, f"roc_auc_run_{run_id:03d}.png")
plt.savefig(IMG_PATH, dpi=200)
plt.close()
print(f"roc_auc plot saved at: {IMG_PATH}")


#f1
plt.figure(figsize=(7,5))
plt.plot(df["epoch"], df["f1"], marker='o', label="f1")
plt.xlabel("Epoch")
plt.ylabel("f1")
plt.title(f"Run {run_id} - f1 per Epoch")
plt.grid(True)
plt.legend()
IMG_PATH = os.path.join(run_folder, f"f1_run_{run_id:03d}.png")
plt.savefig(IMG_PATH, dpi=200)
plt.close()
print(f"f1 plot saved at: {IMG_PATH}")

 
