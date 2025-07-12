import csv
import os

log_path = "loss_history.csv"
header = ['step', 'action_loss', 'policy_loss', 'lam1', 'lam2', 'actor_loss']
write_header = not os.path.exists(log_path)
with open(log_path, 'a', newline='') as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)