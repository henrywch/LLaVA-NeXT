import matplotlib.pyplot as plt
import json

with open('/root/autodl-tmp/models/llavanext-scaled-0.5b/trainer_state.json') as f:
    trainer_state = json.load(f)

steps = []
losses = []

for entry in trainer_state["log_history"]:
    if "loss" in entry:
        steps.append(entry["step"])
        losses.append(entry["loss"])

plt.figure(figsize=(12, 6))
plt.plot(steps, losses, 'b-', linewidth=2)
plt.title('Training Loss Progression', fontsize=14)
plt.xlabel('Training Step', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.yscale('log')
plt.axhline(y=min(losses), color='r', linestyle='--', alpha=0.3)

if len(steps) > 0:
    plt.scatter(steps[0], losses[0], c='red', s=100, label=f'Start: {losses[0]:.4f}')
    plt.scatter(steps[-1], losses[-1], c='green', s=100, label=f'End: {losses[-1]:.4f}')
    
    plt.legend()

output_path = "/root/autodl-tmp/models/llavanext-scaled-0.5b/training_loss.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✅ Loss plot saved to: {output_path}")
