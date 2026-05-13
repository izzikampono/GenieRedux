import os
from pathlib import Path
import hydra
from omegaconf import DictConfig, ListConfig

from tqdm import tqdm

from models import construct_model
from training.evaluation import Evaluator
from training.trainer import collate_with_action_names

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from einops import rearrange
import os
import torch
import torch.nn.functional as F
from data.data import (
    DatasetOutputFormat,
    TransformsGenerator,
    MultiEnvironmentDataset,
    video_tensor_to_gif,
    video_tensor_to_pil_images,
)
from data_generation.generator.utils.retro_act_game_data import GameData
from torch.utils.data import DataLoader

from PIL import Image

from accelerate import Accelerator, DistributedType
from accelerate.utils import DistributedDataParallelKwargs

import logging

logging.basicConfig(level=logging.INFO)
from tools.logger import getLogger

log = getLogger(__name__)


def _normalize_filter(value):
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    if isinstance(value, ListConfig):
        return list(value)
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def resolve_game_whitelist(data_cfg):
    view_filter = _normalize_filter(data_cfg["view"]) if "view" in data_cfg else None
    motion_filter = (
        _normalize_filter(data_cfg["motion"]) if "motion" in data_cfg else None
    )
    genre_filter = _normalize_filter(data_cfg["genre"]) if "genre" in data_cfg else None
    platform_filter = (
        _normalize_filter(data_cfg["platform"]) if "platform" in data_cfg else None
    )

    if (
        view_filter is None
        and motion_filter is None
        and genre_filter is None
        and platform_filter is None
    ):
        return None

    game_data = GameData(
        annotation_fpath=data_cfg["annotation_tag_fpath"],
        control_annotation_fpath=data_cfg.get("annotation_control_fpath"),
        exclude_blinking=bool(data_cfg["exclude_blinking"]),
        exclude_delayed=bool(data_cfg["exclude_delayed"]),
    )
    selected_games = game_data.query(
        view=view_filter,
        motion=motion_filter,
        genre=genre_filter,
        game=None,
        platform=platform_filter,
    )
    if len(selected_games) == 0:
        raise ValueError(
            "No games found for the provided data filters: "
            f"view={view_filter}, motion={motion_filter}, "
            f"genre={genre_filter}, platform={platform_filter}."
        )

    whitelist = [name.lower() for name in selected_games]

    if len(whitelist) == 0:
        raise ValueError(
            "GameData filters produced titles that do not map to dataset directories. "
            "Check annotation entries and dataset naming conventions."
        )

    return whitelist


def get_inference_method(model, args, is_distributed=False):
    if args.model == "tokenizer":
        return model
    if "genie" in args.model:
        if args.eval.inference_method == "autoregressive":
            if is_distributed:
                sample_method = model.generate_interactive_video
            else:
                sample_method = model.module.generate_interactive_video
        elif args.eval.inference_method == "one_go":
            if is_distributed:
                sample_method = model.sample
            else:
                sample_method = model.module.sample

        return sample_method


def convert_index_to_one_hot(index, num_classes):
    one_hot = torch.zeros((*index.shape, num_classes), device=index.device)
    one_hot.scatter_(-1, index.unsqueeze(-1), 1)
    return one_hot


def generate_random_different_action_indices(actions_indices, device, num_actions=7):
    shape = actions_indices.shape
    random_actions = torch.randint(0, num_actions, shape, device=device)

    while torch.any(random_actions == actions_indices):
        random_actions = torch.where(
            random_actions == actions_indices,
            torch.randint(0, num_actions, shape, device=device),
            random_actions,
        )

    return random_actions


def evaluate(
    model,
    evaluator,
    test_loader,
    device,
    args,
    is_main_process=True,
    is_distributed=False,
):
    inference_method = get_inference_method(model, args, is_distributed)
    model_ref = model.module if hasattr(model, "module") else model
    embeddings_required = bool(getattr(model_ref, "use_action_embeddings", False))
    if (
        args.eval.inference_method == "autoregressive"
        and args.eval.sample_num_frames % args.eval.dream_length != 0
    ):
        raise ValueError(
            "Autoregressive eval requires eval.sample_num_frames to be divisible "
            "by eval.dream_length. Otherwise generate_interactive_video produces "
            "a different number of frames than the ground-truth comparison clip. "
            f"Got eval.sample_num_frames={args.eval.sample_num_frames}, "
            f"eval.dream_length={args.eval.dream_length}."
        )
    with torch.no_grad():
        psnr_scores = []
        ssim_scores = []
        delta_psnr_scores = []

        n_previews = max(int(args.eval["n_previews"]), 0)
        reconstruction_enabled = bool(getattr(args.eval, "reconstruction_mode", False))

        for i, batch in enumerate(
            tqdm(
                test_loader,
                total=len(test_loader),
                desc="Evaluating",
                disable=not is_main_process,
            )
        ):

            videos = batch["input_frames"]
            action_names_batch = batch.get("action_name")

            sample_num_frames = args.eval.sample_num_frames
            delta_psnr_horizon = args.eval.delta_psnr_horizon
            num_first_frames = args.eval.num_first_frames
            dream_length = args.eval.dream_length
            num_actions = args.eval.num_actions
            teacher_forcing = bool(getattr(args.eval, "teacher_forcing", False))

            total_frames = num_first_frames + sample_num_frames
            action_horizon = total_frames - 1

            raw_actions = batch["actions"].to(device)[:, :action_horizon]
            num_actions_cfg = int(getattr(args.eval, "num_actions", 0))

            if raw_actions.ndim >= 3:
                inferred_classes = raw_actions.shape[-1]
                actions = raw_actions.argmax(dim=-1).long()
                num_actions = max(num_actions_cfg, inferred_classes)
            else:
                actions = raw_actions.long()
                inferred_classes = (
                    int(actions.max().item()) + 1 if actions.numel() > 0 else 0
                )
                num_actions = max(num_actions_cfg, inferred_classes)
                if num_actions == 0:
                    raise ValueError(
                        "Unable to infer the number of actions from evaluation batch. "
                        "Provide 'args.eval.num_actions'."
                    )

            action_names = None
            if action_names_batch is not None:
                action_names = []
                truncation = num_first_frames + sample_num_frames - 1
                for seq in action_names_batch:
                    if seq is None:
                        action_names.append(None)
                    else:
                        action_names.append(list(seq)[:truncation])

            action_to_take_raw = getattr(args.eval, "action_to_take", -1)
            override_index: int | None = None
            override_name: str | None = None
            override_active = False

            if isinstance(action_to_take_raw, str):
                stripped_value = action_to_take_raw.strip()
                if stripped_value != "":
                    try:
                        override_index = int(stripped_value)
                    except ValueError:
                        override_name = stripped_value
            elif isinstance(action_to_take_raw, (int, float)):
                override_index = int(action_to_take_raw)

            if override_index is not None and override_index >= 0:
                actions = torch.full_like(actions, override_index)
                override_active = True
            elif override_name is not None:
                normalized_override = override_name.strip().lower()
                if embeddings_required:
                    name_to_idx = getattr(model_ref, "_action_name_to_index", {})
                    if normalized_override not in name_to_idx:
                        raise ValueError(
                            f"Override action '{override_name}' not found in model action embeddings."
                        )
                    target_idx = name_to_idx[normalized_override]
                    actions = torch.full_like(actions, target_idx)
                    sequence_len = actions.shape[1]
                    action_names = [
                        [override_name] * sequence_len for _ in range(actions.shape[0])
                    ]
                    override_active = True
                else:
                    if action_names is None or any(seq is None for seq in action_names):
                        raise ValueError(
                            "String-based action overrides require action names to be present in the evaluation batch."
                        )

                    original_actions_list = actions.detach().cpu().tolist()
                    override_indices: list[int] = []
                    updated_action_names: list[list[str]] = []

                    for seq_idx, (seq_names, seq_indices) in enumerate(
                        zip(action_names, original_actions_list)
                    ):
                        name_to_index: dict[str, int] = {}
                        for name, idx in zip(seq_names, seq_indices):
                            if name is None:
                                continue
                            key = str(name).strip().lower()
                            if key not in name_to_index:
                                name_to_index[key] = idx

                        if normalized_override not in name_to_index:
                            raise ValueError(
                                f"Override action '{override_name}' not found among available action names for sample {seq_idx}."
                            )

                        target_idx = name_to_index[normalized_override]
                        override_indices.append(target_idx)
                        updated_action_names.append([override_name] * len(seq_names))

                    actions = torch.tensor(
                        override_indices,
                        device=actions.device,
                        dtype=actions.dtype,
                    ).unsqueeze(1).repeat(1, actions.shape[1])
                    action_names = updated_action_names
                    override_active = True

            if embeddings_required:
                if action_names is None or any(seq is None for seq in action_names):
                    raise ValueError(
                        "Action embeddings are enabled but action names are missing in the evaluation batch. "
                        "Ensure control annotations are provided and collated."
                    )

            videos = videos.to(device)[:, :total_frames]

            videos = rearrange(videos, "b f c h w -> b c f h w")
            ground_truth_sequence = videos.detach().clone()
            reconstruction_video = None

            if reconstruction_enabled and hasattr(model_ref, "dynamics") and hasattr(
                model_ref, "encode_actions"
            ):
                try:
                    video_codebook_ids_full = model_ref.tokenizer(
                        ground_truth_sequence, return_only_codebook_ids=True
                    )
                    video_codebook_ids_full = video_codebook_ids_full.detach()

                    encoded_actions_for_recon = model_ref.encode_actions(
                        actions=actions, action_names=action_names
                    )

                    _, recon_token_flat = model_ref.dynamics(
                        video_codebook_ids=video_codebook_ids_full,
                        actions=encoded_actions_for_recon,
                        return_token_ids=True,
                        tokenizer=model_ref.tokenizer,
                    )

                    target_tokens = video_codebook_ids_full[:, 1:]
                    target_tokens_flat = target_tokens.reshape(target_tokens.shape[0], -1)
                    recon_tokens_flat = recon_token_flat.reshape_as(target_tokens_flat)
                    recon_tokens = recon_tokens_flat.reshape_as(target_tokens)

                    full_recon_tokens = torch.cat(
                        [video_codebook_ids_full[:, :1], recon_tokens], dim=1
                    )

                    reconstruction_video = model_ref.decode_from_codebook_indices(
                        full_recon_tokens
                    )
                    reconstruction_video = torch.clamp(
                        reconstruction_video, min=0.0, max=1.0
                    )
                except Exception as recon_exc:  # noqa: BLE001
                    if is_main_process:
                        log.w(
                            f"Reconstruction mode failed for batch {i}: {type(recon_exc).__name__}: {recon_exc}"
                        )
                    reconstruction_video = None

            first_frames = videos[:, :, :num_first_frames]
            teacher_clip = videos if teacher_forcing else None

            recons = inference_method(
                prime_frames=first_frames,
                actions=actions,
                action_names=action_names,
                num_frames=sample_num_frames,
                inference_steps=args.eval.inference_steps,
                mask_schedule=args.eval.mask_schedule,
                sample_temperature=args.eval.sample_temperature,
                window_size=args.eval.window_size,
                dream_length=dream_length,
                return_recons_only=True,
                teacher_videos=teacher_clip,
            )

            recons = torch.clamp(recons, min=0, max=1)
            videos = videos[:, :, num_first_frames:]
            recons_random = None
            delta_psnr = None
            if args.eval.eval_control and not override_active:
                control_horizon = num_first_frames + delta_psnr_horizon - 1
                action_slice = actions[:, :control_horizon]
                new_actions = action_slice.clone()
                action_names_random = None

                if embeddings_required:
                    if action_names is None or any(seq is None for seq in action_names):
                        raise ValueError(
                            "Control evaluation with action embeddings requires action names."
                        )

                    name_to_idx = getattr(model_ref, "_action_name_to_index", {})
                    known_action_names = list(name_to_idx.keys())
                    if len(known_action_names) < 2:
                        raise ValueError(
                            "Control evaluation requires at least two known action embeddings."
                        )

                    action_names_random = []
                    for sample_idx, seq_names in enumerate(action_names):
                        seq_names_control = list(seq_names[:control_horizon])
                        current_key = model_ref._normalize_action_key(
                            seq_names_control[-1]
                        )
                        candidate_names = [
                            name for name in known_action_names if name != current_key
                        ]
                        if not candidate_names:
                            raise ValueError(
                                "Unable to choose a different action embedding for control evaluation."
                            )

                        replacement_idx = int(
                            torch.randint(
                                len(candidate_names), (1,), device=device
                            ).item()
                        )
                        replacement_name = candidate_names[replacement_idx]
                        seq_names_control[-1] = replacement_name
                        new_actions[sample_idx, -1] = name_to_idx[replacement_name]
                        action_names_random.append(seq_names_control)
                else:
                    random_actions = generate_random_different_action_indices(
                        action_slice,
                        device,
                        num_actions=num_actions,
                    )

                    new_actions[:, -1] = random_actions[:, -1]

                    if action_names is not None:
                        action_names_random = []
                        original_indices_cpu = action_slice.detach().cpu().tolist()
                        new_indices_cpu = new_actions.detach().cpu().tolist()
                        for seq_names, original_indices, new_indices in zip(
                            action_names,
                            original_indices_cpu,
                            new_indices_cpu,
                        ):
                            if seq_names is None:
                                action_names_random.append(None)
                                continue

                            seq_names_control = list(seq_names[:control_horizon])
                            # Build lookup from index to name observed in sequence
                            index_to_name = {
                                idx: name
                                for idx, name in zip(
                                    original_indices, seq_names_control
                                )
                            }
                            replacement_idx = new_indices[-1]
                            replacement_name = index_to_name.get(
                                replacement_idx, str(replacement_idx)
                            )
                            seq_names_control[-1] = replacement_name
                            action_names_random.append(seq_names_control)

                recons_random = inference_method(
                    prime_frames=first_frames,
                    actions=new_actions,
                    action_names=action_names_random,
                    num_frames=delta_psnr_horizon,
                    inference_steps=args.eval.inference_steps,
                    mask_schedule=args.eval.mask_schedule,
                    sample_temperature=args.eval.sample_temperature,
                    window_size=args.eval.window_size,
                    dream_length=dream_length,
                    return_recons_only=True,
                    teacher_videos=teacher_clip,
                )
                recons_random = torch.clamp(recons_random, min=0, max=1)
                delta_psnr = evaluator.delta_psnr(
                    videos[:, :, delta_psnr_horizon - 1 : delta_psnr_horizon],
                    recons[:, :, delta_psnr_horizon - 1 : delta_psnr_horizon],
                    recons_random[:, :, -1:],
                )
                delta_psnr_scores.append(delta_psnr)
            metrics_skipped = override_active
            if metrics_skipped:
                log.i(
                    "Override action provided for this batch; skipping metric aggregation."
                )
            else:
                evaluator.fid_update_batch(videos, recons)

                psnr = evaluator.psnr(videos, recons)
                ssim = evaluator.ssim(
                    rearrange(videos, "b c f h w -> (b f) c h w"),
                    rearrange(recons, "b c f h w -> (b f) c h w"),
                )
                psnr_scores.append(psnr)
                ssim_scores.append(ssim)

                log.i(
                    f"Current scores: {psnr} PSNR; {ssim} SSIM, {delta_psnr} Delta PSNR"
                )

            if i < n_previews and is_main_process:
                tokenizer_recons = None
                if hasattr(model_ref, "tokenizer"):
                    tokenizer_recons = model_ref.tokenizer(
                        ground_truth_sequence, return_recons_only=True
                    )
                    tokenizer_recons = torch.clamp(
                        tokenizer_recons.detach(), min=0.0, max=1.0
                    )

                sampled_videos_path = (
                    Path(args.eval.save_root_dpath)
                    / f"{args.eval.dataset_name}/{args.eval.model_name}/samples"
                )
                sampled_videos_path.mkdir(parents=True, exist_ok=True)
                for j, recons_frames in enumerate(recons.unbind(dim=0)):
                    if j >= n_previews:
                        break
                    orig_frames = torch.cat([first_frames[j], videos[j]], dim=1).detach().cpu()
                    recon_frames = torch.cat([first_frames[j], recons_frames], dim=1).detach().cpu()

                    row_tensors = [orig_frames, recon_frames]
                    if tokenizer_recons is not None:
                        tokenizer_frames = tokenizer_recons[j].detach().cpu()
                        row_tensors.append(tokenizer_frames)
                    if reconstruction_video is not None:
                        reconstruction_frames = reconstruction_video[j].detach().cpu()
                        row_tensors.append(reconstruction_frames)

                    combined_frames = torch.cat(row_tensors, dim=2)

                    pil_rows = [
                        video_tensor_to_pil_images(tensor, only_first_image=False)
                        for tensor in row_tensors
                    ]

                    combined_height = sum(img.height for img in pil_rows)
                    combined_image = Image.new(
                        "RGB", (pil_rows[0].width, combined_height)
                    )

                    offset = 0
                    for img in pil_rows:
                        combined_image.paste(img, (0, offset))
                        offset += img.height

                    video_tensor_to_gif(
                        combined_frames,
                        str(sampled_videos_path / f"sample_{i}_{j}.gif"),
                    )
                    combined_image.save(sampled_videos_path / f"sample_{i}_{j}.png")

            if is_main_process:
                log.i(f"Batch {i} Evaluation done!")

    if psnr_scores:
        psnr_score = torch.mean(torch.tensor(psnr_scores, device=device)).item()
    else:
        psnr_score = float("nan")

    if ssim_scores:
        ssim_score = torch.mean(torch.tensor(ssim_scores, device=device)).item()
    else:
        ssim_score = float("nan")

    if delta_psnr_scores:
        delta_psnr_score = torch.mean(
            torch.tensor(delta_psnr_scores, device=device)
        ).item()
    else:
        delta_psnr_score = float("nan")

    fid_score = evaluator.fid() if psnr_scores else float("nan")

    log.i(
        f"Device: {device},Average FID: {fid_score}, Average PSNR: {psnr_score}, Average SSIM: {ssim_score}, Average Delta PSNR: {delta_psnr_score}"
    )


@torch.no_grad()
def run(args):
    dataset_folder = f"{args.eval.dataset_root_dpath}/{args.eval.dataset_name}"

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[kwargs])

    device = accelerator.device
    evaluator = Evaluator(device)

    model = construct_model(args)

    # Harmonized load behavior with training
    if getattr(args, "model", None) != "tokenizer":
        have_model = bool(getattr(args.eval, "model_fpath", None))
        have_tok = bool(getattr(args, "tokenizer_fpath", None))

        if not have_model and not have_tok:
            raise ValueError("Provide at least one of eval.model_fpath or tokenizer_fpath for evaluation.")

        if have_model and not os.path.exists(args.eval.model_fpath):
            raise FileNotFoundError(
                f"Evaluation model checkpoint not found at '{args.eval.model_fpath}'."
            )
        if have_tok and not os.path.exists(args.tokenizer_fpath):
            raise FileNotFoundError(
                f"Tokenizer checkpoint not found at '{args.tokenizer_fpath}'."
            )

        if have_model:
            model_state_dict = torch.load(args.eval.model_fpath, map_location="cpu")
            model.load_state_dict(model_state_dict["model"])  # strict
            del model_state_dict

        if have_tok:
            tok_state = torch.load(args.tokenizer_fpath, map_location="cpu")
            if not hasattr(model, "tokenizer"):
                raise AttributeError("Model does not expose a 'tokenizer' attribute for loading.")
            model.tokenizer.load_state_dict(tok_state["model"]) 
            del tok_state
    else:
        # Tokenizer-only eval: do not use eval.model_fpath; optional tokenizer_fpath only
        if getattr(args, "tokenizer_fpath", None):
            if not os.path.exists(args.tokenizer_fpath):
                raise FileNotFoundError(
                    f"Tokenizer checkpoint not found at '{args.tokenizer_fpath}'."
                )
            tok_state = torch.load(args.tokenizer_fpath, map_location="cpu")
            model.load_state_dict(tok_state["model"])  # Tokenizer model
            del tok_state

    transforms = TransformsGenerator.get_final_transforms(model.image_size, None)

    game_whitelist = resolve_game_whitelist(args.data)
    if game_whitelist is not None and accelerator.is_main_process:
        log.i(
            f"Evaluation GameData filtering active: {len(game_whitelist)} games selected "
            f"using filters view={args.data['view']}, motion={args.data['motion']}, "
            f"genre={args.data['genre']}, platform={args.data['platform']}."
        )

    test_data_set = MultiEnvironmentDataset(
        dataset_folder,
        seq_length_input=args.eval.num_frames - 1,
        seq_step=args.eval.seq_step,
        split_type="instance",
        split="all",
        transform=transforms["test"],
        format=DatasetOutputFormat.IVG,
        enable_cache=bool(getattr(args.eval, "enable_cache", False)),
        n_workers=args.eval.n_data_workers,
        n_envs=args.eval.n_envs,
        cache_dpath=f"cache/evaluation/{args.eval.dataset_name}",
        whitelist=game_whitelist,
        n_samples=getattr(args.eval, "n_samples", -1),
        annotation_control_fpath=args.data["annotation_control_fpath"],
    )

    test_loader = DataLoader(
        test_data_set,
        batch_size=args.eval.batch_size,
        shuffle=False,
        num_workers=6,
        collate_fn=collate_with_action_names,
    )

    model, test_loader, evaluator = accelerator.prepare(model, test_loader, evaluator)

    is_main_process = accelerator.is_main_process
    is_distributed = accelerator.distributed_type == DistributedType.NO

    evaluate(
        model, evaluator, test_loader, device, args, is_main_process, is_distributed
    )


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
