from __future__ import annotations
import matplotlib
matplotlib.use('Agg')  # Режим без отображения окна (для GitHub Actions)
import matplotlib.pyplot as plt
from pathlib import Path

def generate_delta_plot(result: dict, out_dir: str = "reports/figures"):
    """Строит график падения точности (delta) по типам мутаций."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    agent = result["agents"][result["target_agent"]]
    type_summary = agent["type_summary"]
    
    # Сортируем типы мутаций по убыванию delta (сначала самые опасные)
    sorted_types = sorted(type_summary.items(), key=lambda x: x[1]["mean_delta"], reverse=True)
    
    labels = [k for k, v in sorted_types]
    deltas = [v["mean_delta"] for k, v in sorted_types]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, deltas, color='#d9534f')
    
    plt.title('Деградация качества (Delta) по типам мутаций')
    plt.xlabel('Тип мутации')
    plt.ylabel('Падение качества (0 = нет падения)')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right')
    
    # Подписываем значения над столбцами
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 2), ha='center', va='bottom')
    
    plt.tight_layout()
    file_path = out_path / "delta_by_mutation.png"
    plt.savefig(file_path, dpi=150)
    plt.close()
    return str(file_path)
