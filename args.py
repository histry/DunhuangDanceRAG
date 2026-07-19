import argparse

def add_control_sampling_args(parser):
    parser.add_argument(
        "--beat_guidance_weight",
        type=float,
        default=0.0,
        help="Weight for music beat/onset guidance during inference/TTO",
    )
    parser.add_argument(
        "--hard_keyframe_project",
        action="store_true",
        help="Force strict replacement of known keyframes at every diffusion step",
    )
    return parser

def parse_train_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="runs/train", help="project/name")
    parser.add_argument("--exp_name", default="exp", help="save to project/name")
    parser.add_argument("--data_path", type=str, default="data/dunhuang_bvh/processed", help="path to processed dataset root")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )

    # 修改点 1：将默认特征修改为 hybrid
    parser.add_argument(
        "--feature_type",
        type=str,
        default="hybrid",
        choices=["hybrid", "baseline", "jukebox"],
    )
    # 修改点 2：增加音频特征维度参数，默认 803 (Wav2Vec2 + Librosa)
    parser.add_argument("--audio_dim", type=int, default=803, help="Dimension of the audio feature")
    
    parser.add_argument(
        "--wandb_pj_name", type=str, default="EDGE", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="optimizer learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.02, help="optimizer weight decay")
    parser.add_argument("--seq_len", type=int, default=150, help="motion sequence length")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="accelerate mixed precision mode",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="enable gradient checkpointing in transformer blocks",
    )
    parser.add_argument(
        "--use_sparse_attn",
        action="store_true",
        help="enable local sparse attention mask in decoder self-attention",
    )
    parser.add_argument(
        "--sparse_attn_window",
        type=int,
        default=24,
        help="half-window size for sparse attention (frames)",
    )
    parser.add_argument(
        "--cond_drop_prob",
        type=float,
        default=0.25,
        help="classifier-free guidance drop probability during training",
    )
    parser.add_argument(
        "--audio_pairing_mode",
        type=str,
        default="none",
        choices=["none", "proxy", "paired"],
        help=(
            "Audio-motion pairing mode. "
            "'none' is the safest default for Dunhuang because no real paired "
            "music-motion dataset is available; "
            "'proxy' means weak rhythm proxy only and must not be reported as "
            "paired supervision; "
            "'paired' means verified real paired audio-motion supervision."
        ),
    )
    parser.add_argument(
        "--weak_pairs_path",
        type=str,
        default="data/proxy_weak_pairs/weak_pairs.csv",
        help="CSV file for Dunhuang weak/proxy or paired audio-motion candidates.",
    )

    parser.add_argument(
        "--paired_audio_missing_policy",
        type=str,
        default="error",
        choices=["error", "zero"],
        help=(
            "What to do when --audio_pairing_mode paired but a motion window "
            "has no paired audio candidate. 'error' is safest for strict paired training."
        ),
    )
    parser.add_argument(
        "--mmr_loss_weight",
        type=float,
        default=0.0,
        help=(
            "Weight for MMR audio-motion alignment loss. "
            "Should be >0 only when --audio_pairing_mode paired."
        ),
    )
    parser.add_argument(
        "--keyframe_condition_prob",
        type=float,
        default=0.7,
        help="probability of training with start/end keyframe inpainting conditions; set to 0 to disable",
    )
    parser.add_argument(
        "--keyframe_condition_width",
        type=int,
        default=3,
        help="number of start/end frames exposed as known keyframes during inpainting training",
    )
    parser.add_argument(
        "--trajectory_loss_weight",
        type=float,
        default=1.0,
        help="Weight for supervised root X/Z trajectory position loss.",
    )
    parser.add_argument(
        "--trajectory_velocity_loss_weight",
        type=float,
        default=0.25,
        help="Weight for supervised root X/Z trajectory velocity loss.",
    )

    # ===== TEA-MotionAdapter: energy condition + adapter training =====
    parser.add_argument(
        "--energy_condition_prob",
        type=float,
        default=0.7,
        help="Probability of providing normalized motion energy as a conditioning scalar during training.",
    )
    parser.add_argument(
        "--energy_condition_drop_prob",
        type=float,
        default=0.15,
        help="Drop probability for energy condition only; enables energy-conditioned CFG.",
    )
    parser.add_argument(
        "--energy_loss_weight",
        type=float,
        default=0.25,
        help="Weight for matching generated and target motion-energy envelopes.",
    )

    # ===== V9 RAG Summary Token =====
    parser.add_argument(
        "--enable_rag_summary_token",
        action="store_true",
        help="Enable RAG summary token conditioning.",
    )
    parser.add_argument(
        "--rag_summary_dim",
        type=int,
        default=7,
        help="Dimension of RAG summary vector from rag_context_tokens.py.",
    )
    parser.add_argument(
        "--rag_summary_drop_prob",
        type=float,
        default=0.15,
        help="Drop probability for RAG summary condition during training.",
    )
    parser.add_argument(
        "--root_lower_coupling_loss_weight",
        type=float,
        default=0.5,
        help="Extra multiplier inside kinematic sync loss for root-speed/lower-body coupling.",
    )
    parser.add_argument(
        "--root_lower_speed_threshold",
        type=float,
        default=0.012,
        help="Normalized root XZ speed threshold above which lower-body response is encouraged.",
    )
    parser.add_argument(
        "--root_lower_min_motion",
        type=float,
        default=0.010,
        help="Minimum lower-body rotational motion expected when root XZ speed is high.",
    )
    parser.add_argument(
        "--adapter_train_decoder",
        action="store_true",
        help="With --train_stage adapter, also unfreeze the main seqTransDecoder/final layer.",
    )
    parser.add_argument(
        "--keyframe_loss_weight",
        type=float,
        default=2.0,
        help="loss weight for matching generated motion on exposed keyframe frames",
    )
    parser.add_argument(
        "--contact_loss_weight",
        type=float,
        default=0.8,
        help="extra loss weight for physical 0/1 foot-contact channels; helps avoid all-contact saturation",
    )
    parser.add_argument(
        "--foot_loss_weight",
        type=float,
        default=2.5,
        help="physical foot sliding loss weight after warmup",
    )
    parser.add_argument(
        "--sync_loss_weight",
        type=float,
        default=1.2,
        help="root/leg kinematic sync loss weight after warmup",
    )
    parser.add_argument(
        "--mid_keyframe_condition_prob",
        type=float,
        default=-1.0,
        help="probability of exposing 1-2 middle keyframes during stage2; -1 means auto: 0.7 for stage2, 0 otherwise",
    )
    parser.add_argument(
        "--mid_keyframe_count",
        type=int,
        default=2,
        help="maximum number of middle keyframes to expose per sequence during stage2",
    )
    parser.add_argument(
        "--mid_keyframe_condition_width",
        type=int,
        default=1,
        help="number of frames exposed around each middle keyframe",
    )
    parser.add_argument(
        "--mid_keyframe_selection",
        type=str,
        default="motion_peak",
        choices=["motion_peak", "audio_onset", "mixed", "random"],
        help="how to choose middle keyframes: motion peaks are safest for unpaired Dunhuang motion data",
    )
    parser.add_argument(
        "--dunhuang_split_ratio",
        type=float,
        default=0.9,
        help="source-file split ratio for Dunhuang train/validation data",
    )
    parser.add_argument(
        "--dunhuang_split_seed",
        type=int,
        default=42,
        help="random seed for Dunhuang source-file split",
    )
    parser.add_argument(
        "--dunhuang_val_audio_mode",
        type=str,
        default="best",
        choices=["best", "zero", "random"],
        help="deterministic validation audio selection for weak proxy pairs; random is not recommended for validation",
    )
    parser.add_argument(
        "--traj_aug_prob",
        type=float,
        default=0.3,
        help="Probability of applying geometric trajectory augmentation to Dunhuang motion windows.",
    )
    parser.add_argument(
        "--disable_traj_cond",
        action="store_true",
        help=(
            "Disable trajectory condition during training. "
            "Use this for clean ablation: base/keyframe-only model without X/Z trajectory conditioning."
        ),
    )

    parser.add_argument(
        "--traj_aug_scale_min",
        type=float,
        default=0.8,
        help="Minimum scale for trajectory/root XZ augmentation.",
    )

    parser.add_argument(
        "--traj_aug_scale_max",
        type=float,
        default=1.25,
        help="Maximum scale for trajectory/root XZ augmentation.",
    )

    parser.add_argument(
        "--traj_aug_rot_deg",
        type=float,
        default=30.0,
        help="Maximum absolute rotation angle in degrees for trajectory/root XZ augmentation.",
    )
    parser.add_argument(
        "--train_stage",
        type=str,
        default="full",
        choices=["full", "stage1", "stage2", "adapter"],
        help="stage-wise training shortcut",
    )
    parser.add_argument(
        "--strict_audio_checkpoint",
        action="store_true",
        help=(
            "Raise an error if audio-related checkpoint weights are missing or shape-mismatched. "
            "Use this when you want to claim inherited audio prior."
        ),
    )
    # 🧹 清理：目前网络已默认支持通过传入 Trajectory 字典键名实现动态分支控制
    # 该开关已废弃，暂时注释掉避免与外部传参逻辑冲突
    # parser.add_argument(
    #     "--use_traj_cond",
    #     action="store_true",
    #     help="enable trajectory condition branch (ControlNet-style root guidance)",
    # )
    parser.add_argument(
        "--force_reload", action="store_true", help="force reloads the datasets"
    )
    parser.add_argument(
        "--no_cache", action="store_true", help="don't reuse / cache loaded dataset"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument(
        "--val_batches",
        type=int,
        default=10,
        help="number of validation batches to evaluate at each save interval",
    )

    parser.add_argument(
        "--max_train_batches",
        type=int,
        default=0,
        help="Debug/smoke-test only: stop each epoch after this many training batches. 0 means full epoch.",
    )
    parser.add_argument(
        "--train_num_workers",
        type=int,
        default=-1,
        help="Override training DataLoader workers. -1 keeps default; 0 is best for smoke tests.",
    )
    parser.add_argument(
        "--val_num_workers",
        type=int,
        default=-1,
        help="Override validation DataLoader workers. -1 keeps default; 0 is best for smoke tests.",
    )
    parser.add_argument(
        "--enable_ood_eval",
        action="store_true",
        help="run expensive out-of-distribution music evaluation after checkpoint saves",
    )
    parser.add_argument(
        "--ood_music_dir",
        type=str,
        default="test_music_bank",
        help="directory of OOD wav files used when --enable_ood_eval is set",
    )
    parser.add_argument(
        "--ood_max_files",
        type=int,
        default=0,
        help="limit the number of OOD wav files per evaluation; 0 means all files",
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    add_control_sampling_args(parser)
    opt = parser.parse_args()
    return opt

def parse_test_opt():
    parser = argparse.ArgumentParser()
    
    # 修改点 3：测试时也默认使用 hybrid 特征
    parser.add_argument(
        "--feature_type",
        type=str,
        default="hybrid",
        choices=["hybrid", "baseline", "jukebox"],
    )
    # 修改点 4：测试时同步增加音频特征维度参数
    parser.add_argument("--audio_dim", type=int, default=803, help="Dimension of the audio feature")
    
    parser.add_argument("--out_length", type=float, default=30, help="max. length of output, in seconds")
    parser.add_argument(
        "--processed_data_dir",
        type=str,
        default="data/dataset_backups/",
        help="Dataset backup path",
    )
    parser.add_argument(
        "--render_dir", type=str, default="renders/", help="Sample render path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint.pt", help="checkpoint"
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="data/test/wavs",
        help="folder containing input music",
    )
    parser.add_argument(
        "--save_motions", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        default="eval/motions",
        help="Where to save the motions",
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Save the hybrid features for later reuse", # 修改帮助文档说明
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Don't render the video",
    )
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="cached_features/",
        help="Where to save/load the features",
    )
    parser.add_argument(
        "--use_zero_trajectory",
        action="store_true",
        help="Use a normalized zero XZ trajectory during test generation",
    )
    add_control_sampling_args(parser)
    opt = parser.parse_args()
    return opt
