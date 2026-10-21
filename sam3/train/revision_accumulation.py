"""Image-batch accumulation for new CMPB runs; historical loss rules are unchanged.

The loader supplies 16 images, collated into four interleaved groups of four.
Groups 0/2 and 1/3 reconstruct the two historical rank-local eight-image batches.
Object losses use the historical globally averaged object denominator. CE and
presence use each virtual rank's query denominator and sqrt scale. Contributions
are divided by two to reproduce DDP's rank average, then accumulated. An extra
division by four would incorrectly shrink the gradient. Distributed randomness can still
differ from historical two-GPU runs; numerical identity is not claimed.
"""
import itertools
import json
from pathlib import Path

import torch
from sam3.train.loss.loss_fns import CORE_LOSS_KEY, IABCEMdetr
from sam3.train.loss.sam3_loss import Sam3LossWrapper
from sam3.train.trainer import Trainer


class UpdateClassifier(IABCEMdetr):
    update_context = None

    def get_loss(self, outputs, targets, indices, num_boxes):
        result = super().get_loss(outputs, targets, indices, num_boxes)
        if self.update_context is not None:
            if self.weak_loss or self.use_separate_loss_for_det_and_trk:
                raise ValueError("Only the retained image classification objective is supported")
            weight = len(targets["num_boxes"]) / self.update_context["queries"]
            result["loss_ce"] *= weight
            if "presence_loss" in result:
                result["presence_loss"] *= weight
        return result


class UpdateLoss(Sam3LossWrapper):
    update_context = None

    def __init__(self, *args, reference_ranks=2, **kwargs):
        super().__init__(*args, **kwargs)
        if reference_ranks < 1:
            raise ValueError("reference_ranks must be positive")
        self.reference_ranks = reference_ranks

    def _get_num_boxes(self, targets):
        if self.update_context is None:
            return super()._get_num_boxes(targets)
        return self.update_context["objects"].to(targets["boxes"].device)

    def forward(self, find_stages, find_targets):
        if self.update_context is None:
            return super().forward(find_stages, find_targets)
        if self.loss_fn_semantic_seg is not None or self.normalize_by_stage_num:
            raise ValueError("Only the single-stage CMPB image objective is supported")
        scaled = self.scale_by_find_batch_size
        self.scale_by_find_batch_size = False
        try:
            result = super().forward(find_stages, find_targets)
        finally:
            self.scale_by_find_batch_size = scaled
        if scaled:
            result[CORE_LOSS_KEY] *= self.update_context["queries"] ** 0.5
        result[CORE_LOSS_KEY] /= self.reference_ranks
        return result


class UpdateTrainer(Trainer):
    def _step(self, batch, model, phase):
        contexts = getattr(self, "_update_contexts", None)
        if contexts is not None and phase == "train":
            context = contexts[self._update_chunk]
            self._update_chunk += 1
            loss = self._find_loss("all")
            loss.update_context = context
            for classifier in loss.loss_fns_find:
                if isinstance(classifier, UpdateClassifier):
                    classifier.update_context = context
        return super()._step(batch, model, phase)

    def _run_step(self, batch, phase, loss_mts, extra_loss_mts, raise_on_error=True):
        if torch.distributed.get_world_size() != 1:
            raise ValueError("This configuration is for one GPU per run")
        if not isinstance(batch, list) or len(batch) != self.gradient_accumulation_steps:
            raise ValueError("Expected four collated microbatches per optimizer update")
        if any(set(chunk) != {"all"} for chunk in batch):
            raise ValueError("Expected image-only 'all' batches")
        points = [chunk["all"] for chunk in batch]
        if any(len(point.img_batch) != 4 or len(point.find_targets) != 1 for point in points):
            raise ValueError("Expected four images and one target stage per microbatch")
        loss = self._find_loss("all")
        if not isinstance(loss, UpdateLoss) or loss.normalize_by_valid_object_num:
            raise ValueError("Unexpected loss/normalization configuration")
        targets = [point.find_targets[0] for point in points]
        if loss.reference_ranks != 2:
            raise ValueError("Expected the historical two-rank reference")
        objects = torch.tensor(sum(t.num_boxes.sum().item() for t in targets) / 2,
                               device=self.device, dtype=torch.float).clamp(min=1)
        queries = [sum(len(targets[i].num_boxes) for i in range(rank, 4, 2))
                   for rank in range(2)]
        if min(queries) < 1:
            raise ValueError("Virtual rank has no queries")
        classifiers = [fn for fn in loss.loss_fns_find if isinstance(fn, UpdateClassifier)]
        if len(classifiers) != 1:
            raise ValueError("Expected one accumulated classifier")
        step = self.steps[phase]
        self._update_contexts = [dict(objects=objects, queries=queries[i % 2]) for i in range(4)]
        self._update_chunk = 0
        try:
            return super()._run_step(batch, phase, loss_mts, extra_loss_mts, raise_on_error)
        finally:
            loss.update_context = classifiers[0].update_context = None
            self._update_contexts = None
            # The scheduler and checkpoints count optimizer updates, not forwards.
            self.steps[phase] = step + 1


class UpdateSmokeTrainer(UpdateTrainer):
    def _run_step(self, *args, **kwargs):
        super()._run_step(*args, **kwargs)
        self.peak_allocated = max(getattr(self, "peak_allocated", 0),
                                  torch.cuda.max_memory_allocated())
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
            raise FloatingPointError("Nonfinite or missing smoke-test gradients")
        if not any(torch.count_nonzero(g).item() for g in gradients):
            raise FloatingPointError("All smoke-test gradients are zero")

    def run(self):
        loader = self.train_dataset.get_loader(epoch=int(self.epoch))

        class FirstUpdates:
            def __len__(self):
                return len(loader)

            def __iter__(self):
                return itertools.islice(loader, 3)

        initial = self.steps["train"]
        self.train_epoch(FirstUpdates())
        (Path(self.checkpoint_conf.save_dir).parent / "smoke_result.json").write_text(
            json.dumps({"optimizer_updates": self.steps["train"] - initial,
                        "finite_nonzero_gradients": True,
                        "peak_allocated_bytes": self.peak_allocated,
                        "microbatch_images": 4, "accumulation_steps": 4,
                        "effective_batch_images": 16}) + "\n")
