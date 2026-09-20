"""任务一训练脚本：在服务器上用赛方数据从零训练真实性判别模型。

默认**随机初始化**（``--pretrained-backbone 0``），不依赖任何离线预训练权重，
训练完成后产出的 ``best.pt`` 与 ``task1/scorer.py`` 的推理格式完全兼容。

典型用法::

    # 0) 先体检数据分布（不需要 GPU）
    python -m task1.dataset --data-root /2026aicompetition/datasets/training

    # 1) 从零训练
    python -m task1.train \
        --data-root /2026aicompetition/datasets/training \
        --out-dir  /2026aicompetition/workspace/task1_runs \
        --epochs 40 --batch-size 4 --slices-per-case 16 --image-size 224

    # 2) 让推理服务使用新权重
    export TASK1_WEIGHTS=/2026aicompetition/workspace/task1_runs/best.pt

训练日志按赛方要求写入 ``<log-dir>/training.jsonl``（默认
``$COMPETITION_WORKSPACE/logs``），字段与《赛事开发规范》一致。
"""
from __future__ import annotations

import argparse
import atexit
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .config import DEFAULT_BACKBONE, default_data_root, default_log_dir, default_run_dir
from .dataset import (
    SPECIAL_POLICIES,
    OfficialSlices,
    SliceConfig,
    assign_splits,
    collate,
    discover_records,
    find_annotation_root,
    read_manifest,
    summarize,
    write_manifest,
)
from .metrics import clean_report, metrics_report


def utc_now() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def set_seed(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_logger(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8")
    # 即使训练中途抛异常也要关掉日志文件，避免句柄悬空（Windows 会锁文件）。
    atexit.register(handle.close)

    def log(**fields) -> None:
        handle.write(json.dumps(fields, ensure_ascii=False) + "\n")
        handle.flush()

    return log, handle


def topk_mean(logits, k: int):
    k = max(1, min(k, logits.shape[1]))
    return logits.topk(k, dim=1).values.mean(dim=1)


def forward_cases(model, slices):
    """``(B, K, 3, H, W)`` -> 检查级 logits ``(B,)``：取前一半切片的 logit 均值。

    与推理侧 ``task1/scorer.py`` 的聚合方式严格一致（先平均 logit 再 sigmoid）。
    """
    batch, k = slices.shape[0], slices.shape[1]
    flat = slices.reshape(batch * k, *slices.shape[2:])
    logits = model(flat).reshape(batch, k)
    return topk_mean(logits, max(1, k // 2))


def _autocast(torch, device_type: str, enabled: bool):
    try:
        return torch.amp.autocast(device_type=device_type, enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.0
        return torch.cuda.amp.autocast(enabled=enabled)


def _make_scaler(torch, enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # torch < 2.0
        return torch.cuda.amp.GradScaler(enabled=enabled)


def load_backbone_init(model, path: Path, torch) -> dict:
    """用本地权重文件初始化骨干（可选）。支持 timm 原始权重或 task1 自己的 best.pt。"""
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    backbone_state = {
        key[len("backbone."):]: value
        for key, value in state.items()
        if key.startswith("backbone.")
    }
    if not backbone_state:
        backbone_state = state
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    return {
        "path": str(path),
        "loaded": len(backbone_state),
        "missing": len(missing),
        "unexpected": len(unexpected),
    }


def evaluate(model, loader, device, criterion, torch, *, min_recall: float):
    """返回序列级与检查级指标（检查级 = 同一检查多条序列取最大概率）。"""
    model.eval()
    labels, scores = [], []
    case_scores: dict[str, float] = {}
    case_labels: dict[str, float] = {}
    total_loss, batches = 0.0, 0
    with torch.inference_mode():
        for batch in loader:
            slices = batch["slices"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            logits = forward_cases(model, slices)
            total_loss += float(criterion(logits, target))
            batches += 1
            probabilities = torch.sigmoid(logits).detach().float().cpu().numpy()
            labels.extend(target.detach().float().cpu().numpy().tolist())
            scores.extend(probabilities.tolist())
            for accession, value, label in zip(batch["accession"], probabilities, target.tolist()):
                case_scores[accession] = max(case_scores.get(accession, 0.0), float(value))
                case_labels[accession] = float(label)

    labels_array = np.asarray(labels)
    scores_array = np.asarray(scores)
    case_names = sorted(case_scores)
    case_scores_array = np.asarray([case_scores[name] for name in case_names])
    case_labels_array = np.asarray([case_labels[name] for name in case_names])

    return {
        "loss": total_loss / max(batches, 1),
        "volume_metrics": metrics_report(labels_array, scores_array, min_recall),
        "case_metrics": metrics_report(case_labels_array, case_scores_array, min_recall),
        "cases": len(case_names),
        "volumes": int(labels_array.size),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root(),
                        help="赛方训练数据根目录（含 annotation/）")
    parser.add_argument("--annotation-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="复用已有 manifest.jsonl（跳过目录扫描）")
    parser.add_argument("--out-dir", type=Path, default=default_run_dir())
    parser.add_argument("--log-dir", type=Path, default=None,
                        help="比赛日志目录，默认 $COMPETITION_WORKSPACE/logs 或 <out-dir>/logs")
    parser.add_argument("--composition", choices=SPECIAL_POLICIES, default="exclude")
    parser.add_argument("--duplicate", choices=SPECIAL_POLICIES, default="exclude")
    parser.add_argument("--limit-per-class", type=int, default=0,
                        help="每类只取 N 条序列（调试用，0 = 全部）")

    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--pretrained-backbone", type=int, default=0, choices=(0, 1),
                        help="1 = 用 timm 预训练权重（需要联网或本地缓存），0 = 从零随机初始化")
    parser.add_argument("--backbone-init", type=Path, default=None,
                        help="用本地权重文件初始化骨干（timm 权重或 task1 的 best.pt）")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4, help="每个 batch 的序列数")
    parser.add_argument("--slices-per-case", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4, help="分类头学习率")
    parser.add_argument("--backbone-lr", type=float, default=3e-4, help="骨干学习率（从零训练时通常与头一致）")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=-1.0, help="<0 = 自动取 neg/pos")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-recall", type=float, default=0.5, help="部分 AUC-PR 的召回下界")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", type=int, default=1, choices=(0, 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-frequency-branch", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0, help="每轮最多训练步数（调试用）")
    parser.add_argument("--smoke-test", action="store_true", help="跑几步就退出，验证链路")
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.log_dir or default_log_dir(out_dir)
    log, handle = make_logger(Path(log_dir) / "training.jsonl")

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        from .model import AuthenticityNet
    except Exception as exc:  # noqa: BLE001
        print(f"需要 torch/timm：{type(exc).__name__}: {exc}")
        print("安装：python -m pip install -r task1/requirements.txt")
        return 2

    set_seed(args.seed, torch)
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available())
        else ("cpu" if args.device == "auto" else args.device)
    )

    if args.manifest:
        records = read_manifest(args.manifest)
        annotation_root = None
        data_source = f"{args.manifest.as_posix()}"
    else:
        annotation_root = args.annotation_root or find_annotation_root(args.data_root)
        records = discover_records(
            args.data_root,
            annotation_root=annotation_root,
            composition=args.composition,
            duplicate=args.duplicate,
            limit=args.limit_per_class,
        )
        assign_splits(
            records,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        write_manifest(records, out_dir / "manifest.jsonl")
        data_source = f"{args.data_root.as_posix()}/annotation"

    data_summary = summarize(records)
    print(json.dumps(data_summary, ensure_ascii=False, indent=2), flush=True)
    if data_summary["by_split"].get("val", {}).get("positives", 0) == 0:
        print("[WARN] val 集合没有阳性病例，AP 无法计算；请调大 --val-fraction 或补数据", flush=True)

    train_config = SliceConfig(
        k=args.slices_per_case,
        size=args.image_size,
        train=not args.no_augment,
        slice_mode="random",
    )
    eval_config = SliceConfig(k=args.slices_per_case, size=args.image_size, train=False, slice_mode="uniform")
    train_set = OfficialSlices(records, "train", train_config, seed=args.seed)
    val_set = OfficialSlices(records, "val", eval_config, seed=args.seed)
    if len(train_set) == 0 or len(val_set) == 0:
        print("[ERROR] train/val 至少一个为空，检查数据扫描结果", flush=True)
        return 3

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate, drop_last=False, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate, pin_memory=(device.type == "cuda"),
    )

    pretrained_from = "scratch"
    if args.backbone_init:
        pretrained_from = str(args.backbone_init)
    elif args.pretrained_backbone:
        pretrained_from = f"timm/{args.backbone}"

    model = AuthenticityNet(
        backbone=args.backbone,
        pretrained=bool(args.pretrained_backbone),
        frequency_branch=not args.no_frequency_branch,
    ).to(device)
    if args.backbone_init:
        info = load_backbone_init(model, args.backbone_init, torch)
        print(f"backbone init: {json.dumps(info, ensure_ascii=False)}", flush=True)
    print(f"device={device} backbone={args.backbone} pretrained_from={pretrained_from} "
          f"frequency_branch={not args.no_frequency_branch}", flush=True)

    head_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    backbone_params = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    groups = [{"params": head_params, "lr": args.lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    labels = train_set.labels()
    positives = float(labels.sum())
    negatives = float(labels.size - positives)
    pos_weight = torch.tensor(
        args.pos_weight if args.pos_weight > 0 else max(negatives / max(positives, 1.0), 1.0),
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"train volumes={int(labels.size)} positives={int(positives)} "
          f"neg/pos={negatives / max(positives, 1.0):.2f} pos_weight={float(pos_weight):.2f}", flush=True)

    use_amp = bool(args.amp) and device.type == "cuda"
    scaler = _make_scaler(torch, use_amp)
    data_source_train = f"{data_source}/train"
    data_source_val = f"{data_source}/val"

    best_ap, best_epoch, global_step, stale = -1.0, 0, 0, 0
    started = time.time()
    epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_set.set_epoch(epoch)
        model.train()
        running, seen = 0.0, 0
        for step, batch in enumerate(train_loader, 1):
            if args.max_steps and step > args.max_steps:
                break
            slices = batch["slices"].to(device, non_blocking=True)
            target = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch, device.type, use_amp):
                logits = forward_cases(model, slices)
                loss = criterion(logits, target)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            running += float(loss.detach())
            seen += 1
            global_step += 1
            if step % 10 == 0 or step == 1 or args.smoke_test:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"epoch {epoch} step {step} loss {running / seen:.4f} lr {current_lr:.2e}", flush=True)
                log(timestamp=utc_now(), epoch=epoch, step=global_step, phase="train", mode="training",
                    loss=round(running / seen, 6), lr=current_lr, data_source=data_source_train,
                    checkpoint=None, pretrained_from=pretrained_from)
        scheduler.step()
        train_loss = running / max(seen, 1)

        last_path = out_dir / "last.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": epoch,
                "backbone": args.backbone,
                "image_size": args.image_size,
                "slices_per_case": args.slices_per_case,
                "frequency_branch": not args.no_frequency_branch,
                "val_ap": best_ap,
                "pretrained_from": pretrained_from,
                "data_source": data_source,
                "seed": args.seed,
            },
            last_path,
        )

        metrics = evaluate(model, val_loader, device, criterion, torch, min_recall=args.min_recall)
        case_report = clean_report(metrics["case_metrics"])
        volume_report = clean_report(metrics["volume_metrics"])
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} "
            f"val_case_AP={case_report['average_precision']} "
            f"val_case_partial_AP={case_report['partial_ap']} "
            f"val_case_ROC_AUC={case_report['roc_auc']} "
            f"val_volume_AP={volume_report['average_precision']} "
            f"(cases={metrics['cases']} volumes={metrics['volumes']})",
            flush=True,
        )
        log(
            timestamp=utc_now(), epoch=epoch, step=global_step, phase="val", mode="training",
            loss=round(metrics["loss"], 6), lr=optimizer.param_groups[0]["lr"],
            data_source=data_source_val, checkpoint=str(last_path), pretrained_from=pretrained_from,
            ap=case_report["average_precision"], partial_ap=case_report["partial_ap"],
            roc_auc=case_report["roc_auc"],
        )

        score = case_report["average_precision"]
        score = -1.0 if score is None else float(score)
        if score > best_ap:
            best_ap, best_epoch, stale = score, epoch, 0
            best_path = out_dir / "best.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "backbone": args.backbone,
                    "image_size": args.image_size,
                    "slices_per_case": args.slices_per_case,
                    "frequency_branch": not args.no_frequency_branch,
                    "val_ap": best_ap,
                    "pretrained_from": pretrained_from,
                    "data_source": data_source,
                    "seed": args.seed,
                },
                best_path,
            )
            print(f"  saved {best_path} (val_case_AP={best_ap:.4f})", flush=True)
            log(timestamp=utc_now(), epoch=epoch, step=global_step, phase="val", mode="training",
                loss=round(metrics["loss"], 6), lr=optimizer.param_groups[0]["lr"],
                data_source=data_source_val, checkpoint=str(best_path),
                pretrained_from=pretrained_from, ap=round(best_ap, 6))
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop at epoch {epoch} (best AP {best_ap:.4f} @ epoch {best_epoch})", flush=True)
                break
        if args.smoke_test and global_step >= 2:
            print("smoke test finished", flush=True)
            break

    summary = {
        "data_root": str(args.data_root),
        "annotation_root": None if annotation_root is None else str(annotation_root),
        "data_source": data_source,
        "data_summary": data_summary,
        "backbone": args.backbone,
        "pretrained_from": pretrained_from,
        "frequency_branch": not args.no_frequency_branch,
        "image_size": args.image_size,
        "slices_per_case": args.slices_per_case,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "backbone_lr": args.backbone_lr,
        "pos_weight": float(pos_weight),
        "epochs_run": epoch,
        "best_case_ap": best_ap,
        "best_epoch": best_epoch,
        "minutes": round((time.time() - started) / 60, 2),
        "out_dir": str(out_dir),
        "log_file": str(Path(log_dir) / "training.jsonl"),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
