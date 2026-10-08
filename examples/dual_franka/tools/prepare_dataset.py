"""Create independent EE14 v2.1 train/validation datasets without changing v3 input."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CAMERAS = ["head_image", "left_wrist_image", "right_wrist_image"]
SLICES = {
    "left_ee": (0, 6),
    "right_ee": (6, 12),
    "left_gripper": (12, 13),
    "right_gripper": (13, 14),
}


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    work = args.output.with_name(args.output.name + ".partial")
    work.mkdir(parents=True, exist_ok=True)
    info = json.loads((args.source / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0" and info["fps"] == 30
    assert info["gripper_action_semantics"] == "heuristic_closed_plateau_zero_v1"
    split = json.loads((args.assets / "split.json").read_text())
    episodes = json.loads((args.assets / "episodes.json").read_text())
    assert len(episodes) == 1984
    assert not set(split["train"]) & set(split["validation"])
    assert set(split["train"]) | set(split["validation"]) == set(range(len(episodes)))
    for field, kind in [("observation.state", "ee_pose"), ("action", "delta_ee_pose")]:
        names = [
            f"{side}_{kind}.{a}"
            for side in ["left", "right"]
            for a in ["x", "y", "z", "rx", "ry", "rz"]
        ] + ["left_gripper_width", "right_gripper_width"]
        assert info["features"][field]["names"] == names
    groups = {ep: i for ep, i in [(e["episode_index"], e) for e in episodes]}
    tasks = pq.read_table(args.source / "meta/tasks.parquet").to_pandas().reset_index()
    task_records = [
        {"task_index": int(r["task_index"]), "task": str(r["task"] if "task" in r else r["index"])}
        for r in tasks.to_dict("records")
    ]

    @lru_cache(maxsize=2)
    def table(chunk, file):
        return pq.read_table(
            args.source / info["data_path"].format(chunk_index=chunk, file_index=file)
        )

    jobs = []
    report = {
        "source": str(args.source),
        "split_sha256": hashlib.sha256((args.assets / "split.json").read_bytes()).hexdigest(),
        "partitions": {},
        "video_conversion": "frame-counted lossless H264 re-encode; original videos unchanged",
    }
    for part in ["train", "validation"]:
        root = work / part
        records = []
        total = 0
        for local_idx, ep_id in enumerate(sorted(split[part])):
            ep = groups[ep_id]
            n = int(ep["length"])
            shared = table(ep["data/chunk_index"], ep["data/file_index"])
            mask = pa.compute.equal(shared["episode_index"], ep_id)
            t = shared.filter(mask)
            assert len(t) == n and np.array_equal(t["frame_index"].to_numpy(), np.arange(n))
            for key in ["observation.state", "action"]:
                v = np.stack(t[key].to_pylist())
                assert v.shape == (n, 14) and np.isfinite(v).all()
            t = t.set_column(
                t.schema.get_field_index("episode_index"),
                "episode_index",
                pa.array(np.full(n, local_idx, dtype=np.int64)),
            )
            t = t.append_column("annotation.human.task_description", t["task_index"])
            dst = root / f"data/chunk-{local_idx // 1000:03d}/episode_{local_idx:06d}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, dst)
            # Verify all original numeric fields except intentionally remapped episode_index.
            back = pq.read_table(dst)
            for key in t.column_names:
                assert back[key].equals(t[key]), key
            records.append(
                {
                    "episode_index": local_idx,
                    "length": n,
                    "tasks": ep["tasks"],
                    "source_episode_index": ep_id,
                    "source": ep["source"],
                    "source_episode": ep["source_episode"],
                }
            )
            total += n
            for camera in CAMERAS:
                key = "observation.images." + camera
                src = args.source / info["video_path"].format(
                    video_key=key,
                    chunk_index=ep[f"videos/{key}/chunk_index"],
                    file_index=ep[f"videos/{key}/file_index"],
                )
                dest = (
                    root / f"videos/chunk-{local_idx // 1000:03d}/{key}/episode_{local_idx:06d}.mp4"
                )
                jobs.append((src, dest, float(ep[f"videos/{key}/from_timestamp"]), n))
        out_info = dict(
            info,
            codebase_version="v2.1",
            total_episodes=len(records),
            total_frames=total,
            chunks_size=1000,
            total_chunks=(len(records) + 999) // 1000,
            total_videos=len(records) * 3,
            data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            splits={"train": f"0:{len(records)}"},
        )
        write_json(root / "meta/info.json", out_info)
        (root / "meta/episodes.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records))
        (root / "meta/tasks.jsonl").write_text("".join(json.dumps(x) + "\n" for x in task_records))
        modality = {
            k: {name: {"start": v[0], "end": v[1]} for name, v in SLICES.items()}
            for k in ["state", "action"]
        }
        modality["video"] = {
            k: {"original_key": "observation.images." + v}
            for k, v in zip(["head", "left_wrist", "right_wrist"], CAMERAS)
        }
        modality["annotation"] = {"human.task_description": {}}
        write_json(root / "meta/modality.json", modality)
        report["partitions"][part] = {
            "episodes": len(records),
            "frames": total,
            "valid_window_starts": sum(max(0, r["length"] - 39) for r in records),
            "excluded_tail_starts": sum(min(39, r["length"]) for r in records),
        }

    def convert(job):
        src, dest, start, n = job
        dest.parent.mkdir(parents=True, exist_ok=True)
        marker = dest.with_suffix(".complete.json")
        if (
            marker.exists()
            and dest.exists()
            and json.loads(marker.read_text()) == {"frames": n, "start": start}
        ):
            return
        tmp = dest.with_suffix(".partial.mp4")
        subprocess.run(
            [
                args.ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.9f}",
                "-i",
                str(src),
                "-map",
                "0:v:0",
                "-frames:v",
                str(n),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "0",
                "-threads",
                "1",
                "-y",
                str(tmp),
            ],
            check=True,
        )
        probe = str(Path(args.ffmpeg).with_name("ffprobe"))
        out = subprocess.check_output(
            [
                probe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "json",
                str(tmp),
            ],
            text=True,
        )
        assert int(json.loads(out)["streams"][0]["nb_frames"]) == n
        tmp.rename(dest)
        write_json(marker, {"frames": n, "start": start})

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, _ in enumerate(pool.map(convert, jobs)):
            if i % 100 == 0:
                print(f"video {i}/{len(jobs)}", flush=True)
    write_json(work / "conversion_report.json", report)
    work.rename(args.output)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
