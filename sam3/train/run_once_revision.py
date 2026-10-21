"""Coordinate one existing queued run across an extra GPU and its original queue.

The lock is acquired before constructing the actual trainer/model. A later
invocation waits or skips successful training, leaving evaluation to the queue.
Losses, data and optimization remain in the existing UpdateTrainer.
"""
import fcntl
import json
import os
from pathlib import Path
import time


class CompletedTraining:
    def run(self):
        print("Training already completed by the extra-GPU worker; continuing to evaluation.", flush=True)


class OwnedTraining:
    def __init__(self, trainer, lock, marker, checkpoint):
        self.trainer, self.lock = trainer, lock
        self.marker, self.checkpoint = marker, checkpoint

    def run(self):
        try:
            self.trainer.run()
            if not self.checkpoint.is_file():
                raise RuntimeError("Successful training must produce a Dice-selected checkpoint")
            write_marker(self.marker, "completed")
        except BaseException as exc:
            write_marker(self.marker, "failed", type(exc).__name__)
            raise
        finally:
            self.lock.close()


def write_marker(path, status, error=None):
    value = dict(status=status, pid=os.getpid(), updated_at=time.time(), error_type=error)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def run_once_trainer(**kwargs):
    from sam3.train.revision_accumulation import UpdateTrainer
    return construct_once(UpdateTrainer, kwargs)


def construct_once(trainer_type, kwargs):
    save_dir = Path(kwargs["checkpoint"]["save_dir"])
    directory = save_dir.parent
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "training_once.json"
    checkpoint = save_dir / "val_mean_segmentation_coco_eval_segm_Dice.pt"
    lock = (directory / "training_once.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX)
    if marker.exists():
        status = json.loads(marker.read_text())["status"]
        if status == "completed" and checkpoint.is_file():
            lock.close()
            return CompletedTraining()
        lock.close()
        raise RuntimeError("Previous extra-GPU training failed or ended without a terminal marker; inspect before retrying")
    write_marker(marker, "running")
    try:
        trainer = trainer_type(**kwargs)
    except BaseException as exc:
        write_marker(marker, "failed", type(exc).__name__)
        lock.close()
        raise
    return OwnedTraining(trainer, lock, marker, checkpoint)
