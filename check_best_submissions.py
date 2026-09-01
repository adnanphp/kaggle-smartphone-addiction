import os
import pandas as pd
from pathlib import Path

# Your known AUC values from runs
submissions = {
    'final_optimized_submission.csv': 0.5351,
    'fast_breakthrough_submission.csv': 0.5340,
    'improved_submission.csv': 0.5338,
    'simple_submission_fixed.csv': 0.5310,
    'memory_optimized_submission.csv': 0.5259,
    'optimized_submission.csv': 0.5193,
}

print("="*60)
print("📊 BEST SUBMISSIONS BY VALIDATION AUC")
print("="*60)

# Sort by AUC
sorted_subs = sorted(submissions.items(), key=lambda x: -x[1])

for i, (name, auc) in enumerate(sorted_subs, 1):
    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
    print(f"{medal} {i}. {name}: AUC = {auc:.4f}")

print("\n" + "="*60)
print(f"🏆 BEST SUBMISSION: {sorted_subs[0][0]}")
print(f"   AUC: {sorted_subs[0][1]:.4f}")
print("="*60)

# Check if files exist
print("\n📁 Checking if files exist:")
for name, auc in sorted_subs:
    file_path = Path('submissions') / name
    exists = "✅" if file_path.exists() else "❌"
    print(f"   {exists} {name} ({auc:.4f})")
