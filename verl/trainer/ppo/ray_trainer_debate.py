# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Two-checkpoint debate PPO: model A rolls out first; model B conditions on A's decoded reply
(text sync point); then each party runs a full PPO sub-step on its own worker group.

Constraints (minimal first version):
  - Disable reference-policy KL paths, critic, teacher distillation, REMAX, and rollout_skip.
  - Set ``critic_warmup: 0`` when critic is off so actor updates are not skipped.
  - ``debate.actor_rollout_ref_b`` should mirror ``actor_rollout_ref`` (merge overrides model path, etc.).
  - Optional ``debate.gsm8k.enable``: party A gets a CoT/#### prompt suffix; party B sees A's text; both get
    the same GSM8K score from party B's answer vs ``ground_truth`` (requires ``reward_model.enable=false``).
"""

from __future__ import annotations

import json
import os
import uuid
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.experimental.reward_loop import RewardLoopManager
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.debate_batch_utils import align_batch_repeat_with_output, build_gen_batch_for_party_b
from verl.trainer.ppo.debate_gsm8k import (
    append_party_a_suffix_to_gen_batch,
    build_shared_gsm8k_rm_scores,
    inject_shared_rm_scores,
    merge_debate_cfg_with_gsm8k_defaults,
    shared_gsm8k_metrics,
)
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    Role,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.trainer.ppo.utils import need_reference_policy, need_reward_model, need_teacher_policy
from verl.trainer.ppo.utils_debate import DebateRole
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import log_seqlen_unbalance
from verl.workers.config import CriticConfig, DistillationConfig, EngineConfig


def _debate_party_metric_key(party_name: str, key: str) -> str:
    """Prefix tracking keys so each LLM is a separate W&B group (party_a/*, party_b/*)."""
    return f"party_{party_name}/{key}"


class RayPPODebateTrainer(RayPPOTrainer):
    """PPO trainer with two hybrid actor/rollout groups and a text sync point before party B."""

    def __init__(
        self,
        config,
        tokenizer,
        tokenizer_b,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        processor_b=None,
        train_dataset=None,
        val_dataset=None,
        collate_fn=None,
        train_sampler=None,
        device_name=None,
    ):
        if DebateRole.ActorRolloutB not in role_worker_mapping:
            raise ValueError("Debate trainer requires DebateRole.ActorRolloutB in role_worker_mapping")
        if not OmegaConf.select(config, "debate.actor_rollout_ref_b"):
            raise ValueError("config.debate.actor_rollout_ref_b is required")
        if config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
            raise NotImplementedError("RayPPODebateTrainer does not support REMAX yet")
        if config.actor_rollout_ref.rollout.skip.get("enable", False):
            raise NotImplementedError("Disable rollout.skip for debate training")
        if need_reference_policy(config):
            raise NotImplementedError("RayPPODebateTrainer: turn off reference policy / KL-in-reward for now")
        if need_teacher_policy(config):
            raise NotImplementedError("RayPPODebateTrainer: disable teacher distillation for now")
        if OmegaConf.select(config, "debate.gsm8k.enable", default=False) and need_reward_model(config):
            raise NotImplementedError(
                "debate.gsm8k.shared_reward uses rule-based GSM8K scoring; set reward_model.enable=false."
            )

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            processor=processor,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=device_name,
        )
        self.tokenizer_b = tokenizer_b
        self.processor_b = processor_b

    def _shutdown_dataloaders(self) -> None:
        """Stop DataLoader worker processes before Ray / vLLM teardown (debate fit exit)."""
        for name in ("train_dataloader", "val_dataloader"):
            loader = getattr(self, name, None)
            if loader is None or getattr(loader, "num_workers", 0) == 0:
                continue
            it = getattr(loader, "_iterator", None)
            if it is None:
                continue
            shutdown = getattr(it, "_shutdown_workers", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    pass
            try:
                loader._iterator = None
            except Exception:
                pass

    def _merged_actor_rollout_ref_b(self):
        """Full actor_rollout_ref-style config for party B (A's tree + debate overrides, e.g. ``model.path``)."""
        ar_a = OmegaConf.create(OmegaConf.to_container(self.config.actor_rollout_ref, resolve=True))
        ar_b = OmegaConf.create(OmegaConf.to_container(self.config.debate.actor_rollout_ref_b, resolve=True))
        merged = OmegaConf.merge(ar_a, ar_b)
        # Async vLLM/SGLang HTTP servers register named Ray actors; a second stack must use a distinct prefix.
        rname = OmegaConf.select(merged, "rollout.name", default=None)
        if rname == "vllm":
            with open_dict(merged.rollout):
                merged.rollout.http_ray_server_actor_name_prefix = "vllm_b_"
        elif rname == "vllm_omni":
            with open_dict(merged.rollout):
                merged.rollout.http_ray_server_actor_name_prefix = "vllm_omni_b_"
        elif rname == "sglang":
            with open_dict(merged.rollout):
                merged.rollout.http_ray_server_actor_name_prefix = "sglang_b_"
        return merged

    def _merged_config_party_b(self):
        """Agent loop / rollout replica config for party B (merged over party A)."""
        base = OmegaConf.to_container(self.config, resolve=True)
        merged = OmegaConf.create(base)
        merged.actor_rollout_ref = self._merged_actor_rollout_ref_b()
        return merged

    def init_workers(self):
        """Same as RayPPOTrainer but registers a second actor pool and second agent loop + checkpoint engines."""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                distillation_config=self.config.get("distillation"),
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls

            actor_rollout_resource_pool_b = self.resource_pool_manager.get_resource_pool(DebateRole.ActorRolloutB)
            actor_rollout_cls_b = RayClassWithInitArgs(
                cls=self.role_worker_mapping[DebateRole.ActorRolloutB],
                config=self._merged_actor_rollout_ref_b(),
                distillation_config=self.config.get("distillation"),
                role=str(DebateRole.ActorRolloutB),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool_b][str(DebateRole.ActorRolloutB)] = (
                actor_rollout_cls_b
            )
        else:
            raise NotImplementedError

        orig_critic_cfg = None
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                engine_config: EngineConfig = orig_critic_cfg.engine
                engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            if not class_dict:
                continue
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        self.actor_rollout_wg_b = all_wg[str(DebateRole.ActorRolloutB)]
        self.actor_rollout_wg_b.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
        self.reward_loop_manager = RewardLoopManager(
            config=self.config,
            rm_resource_pool=resource_pool,
        )

        self.async_rollout_mode = True

        if self.use_teacher_policy:
            from verl.experimental.teacher_loop import TeacherModelManager

            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(Role.TeacherModel)
            self.teacher_model_manager = TeacherModelManager(
                config=self.config.distillation,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManagerCls = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            AgentLoopManagerCls = AgentLoopManager

        enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool
        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

        if self.config.actor_rollout_ref.get("hybrid_engine", False) and self.config.actor_rollout_ref.rollout.get(
            "name"
        ) == "vllm":
            # Use the bound RPC (prefixed WorkerDict method), not execute_all_sync(name) on the raw ActorHandle.
            self.actor_rollout_wg.empty_training_cuda_cache()
        self.async_rollout_manager = AgentLoopManagerCls.create(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
            teacher_model_manager=self.teacher_model_manager,
        )

        config_b = self._merged_config_party_b()
        if self.config.actor_rollout_ref.get("hybrid_engine", False) and self.config.actor_rollout_ref.rollout.get(
            "name"
        ) == "vllm":
            self.actor_rollout_wg_b.empty_training_cuda_cache()
        self.async_rollout_manager_b = AgentLoopManagerCls.create(
            config=config_b,
            worker_group=self.actor_rollout_wg_b,
            rollout_resource_pool=actor_rollout_resource_pool_b,
            reward_loop_worker_handles=None,
            teacher_model_manager=self.teacher_model_manager,
        )

        checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=checkpoint_engine_config,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        ce_b = OmegaConf.select(self._merged_actor_rollout_ref_b(), "rollout.checkpoint_engine")
        if ce_b is not None:
            checkpoint_engine_config_b = omega_conf_to_dataclass(ce_b)
        else:
            checkpoint_engine_config_b = checkpoint_engine_config
        self.checkpoint_manager_b = CheckpointEngineManager(
            config=checkpoint_engine_config_b,
            trainer=self.actor_rollout_wg_b,
            replicas=self.async_rollout_manager_b.rollout_replicas,
        )

        self.checkpoint_manager.sleep_replicas()
        self.checkpoint_manager_b.sleep_replicas()

    def _save_checkpoint(self):
        super()._save_checkpoint()

        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        actor_b_local_path = os.path.join(local_global_step_folder, "actor_b")
        actor_b_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor_b")
        )
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        self.actor_rollout_wg_b.save_checkpoint(
            actor_b_local_path, actor_b_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

    def _load_checkpoint(self):
        step = super()._load_checkpoint()
        if self.config.trainer.resume_mode == "disable":
            return step
        base = self.config.trainer.default_local_dir
        if not os.path.isabs(base):
            base = os.path.join(os.getcwd(), base)
        global_step_folder = os.path.join(base, f"global_step_{self.global_steps}")
        actor_b_path = os.path.join(global_step_folder, "actor_b")
        if os.path.exists(actor_b_path):
            self.actor_rollout_wg_b.load_checkpoint(
                actor_b_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )
        else:
            print(f"Warning: no actor_b checkpoint at {actor_b_path}; party B starts from init weights")
        return step

    def _start_profiling(self, do_profile: bool) -> None:
        super()._start_profiling(do_profile)
        if do_profile:
            self.actor_rollout_wg_b.start_profile(role="e2e", profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        super()._stop_profiling(do_profile)
        if do_profile:
            self.actor_rollout_wg_b.stop_profile()

    def _collect_debate_trajectory_rows(self, batch_a: DataProto, batch_b: DataProto) -> list[dict]:
        """One dict per row: prompts/responses/rewards/metadata (shared by JSONL and W&B)."""
        n = len(batch_a)
        if len(batch_b) != n:
            raise ValueError(f"debate trajectory dump: batch_a len {n} != batch_b len {len(batch_b)}")

        prompts_a = self.tokenizer.batch_decode(batch_a.batch["prompts"], skip_special_tokens=True)
        responses_a = self.tokenizer.batch_decode(batch_a.batch["responses"], skip_special_tokens=True)
        prompts_b = self.tokenizer_b.batch_decode(batch_b.batch["prompts"], skip_special_tokens=True)
        responses_b = self.tokenizer_b.batch_decode(batch_b.batch["responses"], skip_special_tokens=True)

        r_a = batch_a.batch["token_level_scores"].sum(-1).detach().cpu()
        r_b = batch_b.batch["token_level_scores"].sum(-1).detach().cpu()

        def _scalar(x):
            if hasattr(x, "item"):
                return x.item()
            return x

        rows: list[dict] = []
        for i in range(n):
            rm = batch_a[i].non_tensor_batch.get("reward_model", {}) or {}
            gt = rm.get("ground_truth", None)
            if hasattr(gt, "tolist"):
                gt = gt.tolist()
            ds = batch_a[i].non_tensor_batch.get("data_source", None)
            if ds is not None and not isinstance(ds, str):
                ds = str(ds)
            uid = batch_a[i].non_tensor_batch.get("uid", None)
            if uid is not None and not isinstance(uid, str):
                uid = str(uid)
            rows.append(
                {
                    "global_step": int(self.global_steps),
                    "uid": uid,
                    "data_source": ds,
                    "ground_truth": gt,
                    "party_a_prompt": prompts_a[i],
                    "party_a_response": responses_a[i],
                    "party_b_prompt": prompts_b[i],
                    "party_b_response": responses_b[i],
                    "reward_party_a": _scalar(r_a[i]),
                    "reward_party_b": _scalar(r_b[i]),
                }
            )
        return rows

    def _balance_debate_batches_in_lockstep(self, batch_a: DataProto, batch_b: DataProto, metrics: dict) -> None:
        """Apply the same DP-balancing permutation to both parties (from party A's lengths).

        Per-party ``_balance_batch`` uses each batch's own sequence lengths, so A and B get
        different permutations and row ``i`` can refer to different problems — breaking JSONL,
        shared GSM8K scores, and any index-aligned pairing.
        """
        global_idx, global_partition_lst, seqlen_list = self._balance_batch_global_idx(
            batch_a, keep_minibatch=False
        )
        batch_a.reorder(global_idx)
        batch_b.reorder(global_idx)
        metrics.update(
            log_seqlen_unbalance(
                seqlen_list=seqlen_list, partitions=global_partition_lst, prefix="global_seqlen_debate"
            )
        )

    def _assert_debate_batches_row_aligned(self, batch_a: DataProto, batch_b: DataProto) -> None:
        """Fail fast if party A and B rows diverge (wrong rewards, JSONL, shared GSM8K)."""
        n, m = len(batch_a), len(batch_b)
        if n != m:
            raise ValueError(f"debate batches length mismatch: party_a={n} party_b={m}")
        uids_a = batch_a.non_tensor_batch.get("uid")
        uids_b = batch_b.non_tensor_batch.get("uid")
        if uids_a is None or uids_b is None:
            return
        for i in range(n):
            if uids_a[i] != uids_b[i]:
                raise ValueError(
                    f"Debate trainer: row {i} uid mismatch (party_a={uids_a[i]!r}, party_b={uids_b[i]!r}). "
                    "Per-party reordering or a bug broke index alignment; rewards would be wrong."
                )

    def _log_debate_joint_trajectories_jsonl(self, rows: list[dict], filepath: str) -> None:
        """Append JSONL (one object per line)."""
        path = os.path.expanduser(filepath)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        with open(path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

        print(f"Appended {len(rows)} debate trajectory row(s) to {path}")

    def _log_debate_trajectories_wandb(self, rows: list[dict], tracking) -> None:
        """Log a ``wandb.Table`` of trajectories when W&B (or veMLP W&B) is enabled."""
        if not self.config.trainer.get("debate_wandb_log_trajectories", False):
            return
        if not rows:
            return

        wandb_mod = tracking.logger.get("wandb") or tracking.logger.get("vemlp_wandb")
        if wandb_mod is None:
            return
        run = getattr(wandb_mod, "run", None)
        if run is None:
            return

        max_rows = int(self.config.trainer.get("debate_wandb_trajectory_max_rows", 32))
        max_chars = self.config.trainer.get("debate_wandb_trajectory_max_chars", 4000)

        def _clip(cell):
            if cell is None:
                return ""
            s = str(cell)
            if max_chars is None or len(s) <= max_chars:
                return s
            return s[: max_chars - 3] + "..."

        columns = list(rows[0].keys())
        table = wandb_mod.Table(columns=columns)
        for row in rows[:max_rows]:
            table.add_data(*[_clip(row.get(c)) for c in columns])

        wandb_mod.log({"debate/trajectories": table}, step=int(self.global_steps))

    def _execute_party_ppo(
        self,
        batch: DataProto,
        metrics: dict,
        timing_raw: dict,
        party_name: str,
        wg,
        ckpt_mgr,
        async_mgr,
        skip_balance: bool = False,
    ) -> tuple[DataProto, dict]:
        """One full PPO forward/backward for a single party (reward → adv → actor update → weight sync)."""
        saved_wg = self.actor_rollout_wg
        saved_ckpt = self.checkpoint_manager
        saved_async = self.async_rollout_manager
        self.actor_rollout_wg = wg
        self.checkpoint_manager = ckpt_mgr
        self.async_rollout_manager = async_mgr
        reward_extra_infos_dict: dict = {}
        try:
            if self._should_compute_teacher_colocate(batch):
                with marked_timer(f"teacher/{party_name}", timing_raw, color="cyan"):
                    batch_teacher = self._compute_teacher_colocate(batch)
                    batch = batch.union(batch_teacher)

            if "response_mask" not in batch.batch.keys():
                batch.batch["response_mask"] = compute_response_mask(batch)
            if self.config.trainer.balance_batch and not skip_balance:
                self._balance_batch(batch, metrics=metrics)

            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
            images_seqlens_all = []
            if "multi_modal_inputs" in batch.non_tensor_batch:
                for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                    if "image_grid_thw" not in multi_modal_input.keys():
                        continue
                    images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
            batch.meta_info["images_seqlens"] = images_seqlens_all

            with marked_timer(f"reward/{party_name}", timing_raw, color="yellow"):
                if "rm_scores" not in batch.batch.keys():
                    batch_reward = self._compute_reward_colocate(batch)
                    batch = batch.union(batch_reward)
                reward_tensor, reward_extra_infos_dict = extract_reward(batch)

            rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
            bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
            if bypass_recomputing_logprobs:
                from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                apply_bypass_mode(
                    batch=batch,
                    rollout_corr_config=rollout_corr_config,
                    policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                )
            else:
                with marked_timer(f"old_log_prob/{party_name}", timing_raw, color="blue"):
                    old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    actor_config = self.config.actor_rollout_ref.actor
                    entropy_agg = agg_loss(
                        loss_mat=entropys,
                        loss_mask=response_masks,
                        loss_agg_mode=actor_config.loss_agg_mode,
                        loss_scale_factor=actor_config.loss_scale_factor,
                    )
                    metrics[_debate_party_metric_key(party_name, "actor/entropy")] = entropy_agg.detach().item()
                    metrics[_debate_party_metric_key(party_name, "perf/mfu/actor_infer")] = old_log_prob_mfu
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)
                    if "rollout_log_probs" in batch.batch.keys():
                        from verl.utils.debug.metrics import calculate_debug_metrics

                        metrics.update(
                            {
                                _debate_party_metric_key(party_name, k): v
                                for k, v in calculate_debug_metrics(batch).items()
                            }
                        )

            assert "old_log_probs" in batch.batch

            if self.use_reference_policy:
                with marked_timer(f"{Role.RefPolicy}/{party_name}", timing_raw, color="olive"):
                    ref_log_prob = self._compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            if self.use_critic:
                with marked_timer(f"values/{party_name}", timing_raw, color="cyan"):
                    values = self._compute_values(batch)
                    batch = batch.union(values)

            with marked_timer(f"adv/{party_name}", timing_raw, color="brown"):
                batch.batch["token_level_scores"] = reward_tensor
                if reward_extra_infos_dict:
                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                if self.config.algorithm.use_kl_in_reward:
                    batch, kl_metrics = apply_kl_penalty(
                        batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                    )
                    metrics.update(
                        {_debate_party_metric_key(party_name, k): v for k, v in kl_metrics.items()}
                    )
                else:
                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                if (
                    rollout_corr_config is not None
                    and "rollout_log_probs" in batch.batch
                    and not bypass_recomputing_logprobs
                ):
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                    metrics.update(
                        {_debate_party_metric_key(party_name, k): v for k, v in is_metrics.items()}
                    )

                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                batch = compute_advantage(
                    batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    config=self.config.algorithm,
                )

            if self.use_critic:
                with marked_timer(f"update_critic/{party_name}", timing_raw, color="pink"):
                    critic_output = self._update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(
                    {_debate_party_metric_key(party_name, k): v for k, v in critic_output_metrics.items()}
                )

            if self.config.trainer.critic_warmup > self.global_steps:
                ckpt_mgr.update_weights(self.global_steps)
            else:
                with marked_timer(f"update_actor/{party_name}", timing_raw, color="red"):
                    actor_output = self._update_actor(batch)
                with marked_timer(f"update_weights/{party_name}", timing_raw, color="red"):
                    ckpt_mgr.update_weights(self.global_steps)
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(
                    {_debate_party_metric_key(party_name, k): v for k, v in actor_output_metrics.items()}
                )

            return batch, reward_extra_infos_dict
        finally:
            self.actor_rollout_wg = saved_wg
            self.checkpoint_manager = saved_ckpt
            self.async_rollout_manager = saved_async

    def fit(self):
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)
        self.checkpoint_manager_b.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                for wg in (self.actor_rollout_wg, self.actor_rollout_wg_b):
                    if hasattr(wg, "async_calls_finalize_fn_exec"):
                        wg.async_calls_finalize_fn_exec(blocking=False)

                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                debate_cfg = merge_debate_cfg_with_gsm8k_defaults(self.config.debate)
                if OmegaConf.select(debate_cfg, "gsm8k.enable", default=False):
                    party_a_suffix = OmegaConf.select(debate_cfg, "gsm8k.party_a_suffix", default="")
                    gen_batch_output = append_party_a_suffix_to_gen_batch(gen_batch_output, party_a_suffix)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    with marked_timer("gen_party_a", timing_raw, color="red"):
                        if curr_step_profile:
                            self.async_rollout_manager.start_profile()
                        gen_out_a = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        self.checkpoint_manager.sleep_replicas()
                        if curr_step_profile:
                            self.async_rollout_manager.stop_profile()
                        timing_raw.update(gen_out_a.meta_info["timing"])
                        gen_out_a.meta_info.pop("timing", None)

                    device = gen_out_a.batch["prompts"].device
                    gen_b_in = build_gen_batch_for_party_b(
                        output_party_a=gen_out_a,
                        tokenizer_a=self.tokenizer,
                        tokenizer_b=self.tokenizer_b,
                        debate_cfg=debate_cfg,
                        max_prompt_length=self.config.data.max_prompt_length,
                        device=device,
                    )
                    for k, v in gen_batch.meta_info.items():
                        gen_b_in.meta_info.setdefault(k, v)

                    with marked_timer("gen_party_b", timing_raw, color="red"):
                        if curr_step_profile:
                            self.async_rollout_manager_b.start_profile()
                        gen_out_b = self.async_rollout_manager_b.generate_sequences(gen_b_in)
                        self.checkpoint_manager_b.sleep_replicas()
                        if curr_step_profile:
                            self.async_rollout_manager_b.stop_profile()
                        timing_raw.update(gen_out_b.meta_info["timing"])
                        gen_out_b.meta_info.pop("timing", None)

                    n = self.config.actor_rollout_ref.rollout.n
                    batch_a = align_batch_repeat_with_output(batch, gen_out_a, n)
                    batch_b = align_batch_repeat_with_output(batch, gen_out_b, n)

                    # Balance before shared reward: same permutation on both batches, then score the
                    # final row order so rm_scores / ground_truth / responses stay aligned for PPO.
                    if self.config.trainer.balance_batch:
                        self._balance_debate_batches_in_lockstep(batch_a, batch_b, metrics)

                    if OmegaConf.select(debate_cfg, "gsm8k.enable", default=False):
                        gsk = debate_cfg.gsm8k
                        shared_scores = build_shared_gsm8k_rm_scores(
                            batch_b,
                            self.tokenizer_b,
                            method=gsk.get("method", "strict"),
                            format_score=float(gsk.get("format_score", 0.0)),
                            score=float(gsk.get("score", 1.0)),
                        )
                        inject_shared_rm_scores(batch_a, shared_scores)
                        inject_shared_rm_scores(batch_b, shared_scores)
                        metrics.update(shared_gsm8k_metrics(shared_scores))

                    self._assert_debate_batches_row_aligned(batch_a, batch_b)

                    batch_a, reward_extra_a = self._execute_party_ppo(
                        batch_a,
                        metrics,
                        timing_raw,
                        "a",
                        self.actor_rollout_wg,
                        self.checkpoint_manager,
                        self.async_rollout_manager,
                        skip_balance=True,
                    )
                    batch_b, reward_extra_b = self._execute_party_ppo(
                        batch_b,
                        metrics,
                        timing_raw,
                        "b",
                        self.actor_rollout_wg_b,
                        self.checkpoint_manager_b,
                        self.async_rollout_manager_b,
                        skip_balance=True,
                    )

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch_a, reward_extra_a, timing_raw, rollout_data_dir)
                        self._log_rollout_data(batch_b, reward_extra_b, timing_raw, rollout_data_dir)

                    traj_path = self.config.trainer.get("debate_trajectory_jsonl", None)
                    want_wandb_traj = self.config.trainer.get("debate_wandb_log_trajectories", False)
                    if traj_path or want_wandb_traj:
                        with marked_timer("dump_debate_trajectories", timing_raw, color="green"):
                            traj_rows = self._collect_debate_trajectory_rows(batch_a, batch_b)
                            if traj_path:
                                self._log_debate_joint_trajectories_jsonl(traj_rows, traj_path)
                            if want_wandb_traj:
                                self._log_debate_trajectories_wandb(traj_rows, logger)

                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                batch = batch_b
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                for bn, b in (("a", batch_a), ("b", batch_b)):
                    dm = compute_data_metrics(batch=b, use_critic=self.use_critic)
                    metrics.update({_debate_party_metric_key(bn, k): v for k, v in dm.items()})
                gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
                if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
                    for bn, b in (("a", batch_a), ("b", batch_b)):
                        for key in gdpo_reward_keys:
                            if key in b.non_tensor_batch:
                                vals = np.asarray(b.non_tensor_batch[key], dtype=np.float32)
                                metrics[_debate_party_metric_key(bn, f"gdpo/{key}/mean")] = float(np.mean(vals))
                                metrics[_debate_party_metric_key(bn, f"gdpo/{key}/std")] = float(np.std(vals))
                                metrics[_debate_party_metric_key(bn, f"gdpo/{key}/max")] = float(np.max(vals))
                                metrics[_debate_party_metric_key(bn, f"gdpo/{key}/min")] = float(np.min(vals))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                total_tok = sum(batch_a.meta_info["global_token_num"]) + sum(batch_b.meta_info["global_token_num"])
                step_t = timing_raw["step"]
                metrics.update(
                    {
                        "perf/total_num_tokens": total_tok,
                        "perf/time_per_step": step_t,
                        "perf/throughput": total_tok / (step_t * n_gpus) if step_t > 0 else 0.0,
                    }
                )
                for bn, b in (("a", batch_a), ("b", batch_b)):
                    gnorm = metrics.get(_debate_party_metric_key(bn, "actor/grad_norm"), None)
                    vm = compute_variance_proxy_metrics(batch=b, gradient_norm=gnorm)
                    metrics.update({_debate_party_metric_key(bn, k): v for k, v in vm.items()})

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    for wg in (self.actor_rollout_wg, self.actor_rollout_wg_b):
                        wg.dump_memory_snapshot(
                            tag=f"post_update_step{self.global_steps}_{id(wg)}",
                            sub_dir=f"step{self.global_steps}",
                        )

                if is_last_step:
                    for wg in (self.actor_rollout_wg, self.actor_rollout_wg_b):
                        if hasattr(wg, "async_calls_finalize_fn_exec"):
                            wg.async_calls_finalize_fn_exec(blocking=True)
                    self._shutdown_dataloaders()
                    if self.config.trainer.test_freq > 0:
                        pprint(f"Final validation metrics: {last_val_metrics}")
                    else:
                        print(
                            "Skipping periodic validation (trainer.test_freq<=0); "
                            "no final validation metrics to report."
                        )
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
