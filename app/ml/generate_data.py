import pandas as pd
import numpy as np
from pathlib import Path

N = 50000  # можно сделать 50k, 100k и т.д.
rng = np.random.default_rng(42)

# случайные значения
distance = rng.uniform(0, 100, N)
residency = rng.uniform(0, 10, N)
is_immigrant = rng.choice([True, False], N)
parents_status = rng.choice(["low", "middle", "high"], N)

# добавим дубли (примерно 10%)
dup_idx = rng.choice(N, N // 10)
distance = np.concatenate([distance, distance[dup_idx]])
residency = np.concatenate([residency, residency[dup_idx]])
is_immigrant = np.concatenate([is_immigrant, is_immigrant[dup_idx]])
parents_status = np.concatenate([parents_status, parents_status[dup_idx]])

# собираем датафрейм
df = pd.DataFrame({
    "distance_km": np.round(distance, 1),
    "residency_years_ie": np.round(residency, 1),
    "is_immigrant": is_immigrant,
    "parents_status": parents_status
})

# сохраняем
out_path = Path("data/raw/generated_applicants.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_path, index=False)

print(f"✅ Сохранено {len(df):,} строк в {out_path}")
print(df.head())
