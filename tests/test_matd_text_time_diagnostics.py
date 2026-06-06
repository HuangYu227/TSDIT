import numpy as np
import pandas as pd

from mmldm.matd.text_time_diagnostics import compute_text_time_diagnostics


def test_text_time_diagnostics_basic_dataframe():
    rows = []
    for i in range(30):
        level = i % 3
        text = f"level {level}"
        ot = (np.linspace(0, 1, 8) + float(level)).tolist()
        emb = [float(level), float(level == 1), float(level == 2)]
        rows.append(
            {
                "Text": text,
                "TextEmbedding": str(emb),
                "OT": str(ot),
            }
        )
    results = compute_text_time_diagnostics(pd.DataFrame(rows))
    assert results["sample_count"] == 30.0
    assert results["unique_text_count"] == 3.0
    assert results["text_embedding_dim"] == 3.0
    assert np.isfinite(results["text_stat_r2_mean"])
    assert "duplicate_text_stat_std_ratio_mean" in results
