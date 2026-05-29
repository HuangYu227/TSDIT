"""Convert CaTSG datasets to TIGER CSV format.

Output: embedding_cleaned_{dataset}_{length}.csv
Columns: SampleID, SampleNumID, TimeInterval, Text, TextEmbedding, OT
"""

import os
import argparse
import numpy as np
import pandas as pd


def describe_aq(sample_c):
    """Generate text description for Air Quality sample.
    c: (T, 6) -> [TEMP, PRES, DEWP, WSPM, RAIN, wd]
    """
    means = sample_c.mean(axis=0)
    temp, pres, dewp, wspm, rain, wd = means

    wd_labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    wd_idx = int(round(wd))
    wd_str = wd_labels[wd_idx] if 0 <= wd_idx < 15 else 'variable'

    def qual(v, thresholds):
        for th, label in thresholds:
            if v < th:
                return label
        return thresholds[-1][1]

    temp_str = qual(temp, [(-0.5, "cold"), (0.0, "cool"), (0.5, "mild"), (99, "warm")])
    wind_str = qual(wspm, [(-0.3, "light wind"), (0.3, "moderate wind"), (99, "strong wind")])
    rain_str = qual(rain, [(-0.2, "dry conditions"), (0.5, "slight precipitation"), (99, "rainy")])

    return f"PM2.5 air quality reading with {temp_str} temperature, {wind_str} from {wd_str}, and {rain_str}."


def describe_traffic(sample_c):
    """Generate text description for Traffic sample.
    c: (T, 5) -> [rain_1h, snow_1h, clouds_all, weather_main, holiday]
    """
    means = sample_c.mean(axis=0)
    rain, snow, clouds, weather, holiday = means

    weather_labels = ['Clear', 'Clouds', 'Rain', 'Drizzle', 'Mist', 'Haze',
                      'Fog', 'Thunderstorm', 'Snow', 'Squall', 'Smoke']
    weather_idx = int(round(weather))
    weather_str = weather_labels[weather_idx] if 0 <= weather_idx < 10 else 'mixed'

    is_holiday = holiday > 0.5
    holiday_str = "holiday period" if is_holiday else "regular day"

    def qual(v, thresholds):
        for th, label in thresholds:
            if v < th:
                return label
        return thresholds[-1][1]

    cloud_str = qual(clouds, [(-0.5, "clear skies"), (0.0, "partly cloudy"),
                              (0.5, "mostly cloudy"), (99, "overcast")])
    rain_str = qual(rain, [(-0.3, "no rain"), (0.3, "light rain"), (99, "heavy rain")])

    return f"Traffic volume during a {holiday_str} with {weather_str.lower()} weather, {cloud_str}, and {rain_str}."


def convert(catsg_dir, output_dir, generate_embeddings=False):
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        'aq': {
            'subdir': 'station_based',
            'describe_fn': describe_aq,
            'tiger_name': 'catsg_aq',
        },
        'traffic': {
            'subdir': 'temp_based',
            'describe_fn': describe_traffic,
            'tiger_name': 'catsg_traffic',
        },
    }

    # Lazy-load CLIP only if needed
    clip_model, clip_tokenizer, clip_device = None, None, None
    if generate_embeddings:
        import torch
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        clip_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clip_tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        clip_model = CLIPTextModelWithProjection.from_pretrained('openai/clip-vit-base-patch32').to(clip_device)
        clip_model.eval()

    def embed_texts(texts, batch_size=256):
        all_embs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = clip_tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(clip_device) for k, v in inputs.items()}
            with torch.no_grad():
                out = clip_model(**inputs)
            all_embs.append(out.text_embeds.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    for ds_name, ds_cfg in datasets.items():
        print(f"\n=== {ds_name} ===")
        data_dir = os.path.join(catsg_dir, ds_name, ds_cfg['subdir'])
        if not os.path.exists(data_dir):
            print(f"  [SKIP] {data_dir} not found")
            continue

        # Collect all splits into single CSV (TIGER splits in code)
        all_rows = []
        split_counts = {}
        sample_id = 0

        for split in ['train', 'val', 'test']:
            x_path = os.path.join(data_dir, f"x_{split}.npy")
            c_path = os.path.join(data_dir, f"c_{split}.npy")
            if not os.path.exists(x_path):
                continue

            x = np.load(x_path)  # (N, T, 1)
            c = np.load(c_path)  # (N, T, D)
            N, T, _ = x.shape
            x_flat = x.squeeze(-1)  # (N, T)

            for i in range(N):
                text = ds_cfg['describe_fn'](c[i])
                all_rows.append({
                    'SampleID': sample_id,
                    'SampleNumID': sample_id,
                    'TimeInterval': T,
                    'Text': text,
                    'TextEmbedding': '',  # filled below if requested
                    'OT': str(x_flat[i].tolist()),
                    'split': split,  # preserve original split
                })
                sample_id += 1

            split_counts[split] = N
            print(f"  [{split}] {N} samples loaded")

        if not all_rows:
            print("  [SKIP] no data found")
            continue

        df = pd.DataFrame(all_rows)

        # Generate CLIP embeddings if requested
        if generate_embeddings:
            print(f"  Generating CLIP embeddings for {len(df)} texts...")
            embs = embed_texts(df['Text'].tolist())
            df['TextEmbedding'] = [np.array2string(e, separator=' ') for e in embs]

        out_path = os.path.join(output_dir, f"embedding_cleaned_{ds_cfg['tiger_name']}_96.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path} ({len(df)} samples)")
        print(f"  Split counts: {split_counts}")

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--catsg_dir', default='E:/Research/TSG/CaTSG/dataset')
    parser.add_argument('--output_dir', default='E:/Research/TSG/myTSG_V0/Three Levels Data/CaTSG')
    parser.add_argument('--generate_embeddings', action='store_true')
    args = parser.parse_args()
    convert(args.catsg_dir, args.output_dir, args.generate_embeddings)
