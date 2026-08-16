"""Runtime configuration for the reproducible GraphReason-Bot package.

No credentials are stored in this repository.  Supply provider credentials via
environment variables or a local ``.env`` file that is excluded by git.
"""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]
try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    # Environment variables remain supported when python-dotenv is omitted.
    pass
DATA_ROOT = Path(
    os.environ.get("GRAPHREASON_DATA_ROOT", str(REPO_ROOT / "data"))
).expanduser()
RESULT_DIR = Path(
    os.environ.get("GRAPHREASON_RESULTS_DIR", str(REPO_ROOT / "outputs"))
).expanduser()

# Expected preprocessed dataset locations.  Datasets are intentionally not
# versioned here; consult data/README.md for provenance and preparation.
DATASET_PATHS = {
    "BotSim-24": DATA_ROOT / "BotSim-24" / "processed" / "test.json",
    "Twibot-22": DATA_ROOT / "Twibot-22" / "community_format" / "test.json",
}
SELECTED_1000_PATHS = {
    "BotSim-24": DATA_ROOT / "BotSim-24" / "selected_1000.json",
    "Twibot-22": DATA_ROOT / "Twibot-22" / "selected_1000.json",
}
DATASETS = ["BotSim-24", "Twibot-22"]
DATA_PATH = DATASET_PATHS["Twibot-22"]

# Safe defaults for command-line runs.  Expensive analyses are opt-in through
# their dedicated commands in docs/EXPERIMENTS.md.
PYCHARM_EVAL_DATASET = "all"
PYCHARM_EVAL_DATA_SOURCE = "fixed_test"
PYCHARM_PARALLEL_DATASETS = False
PYCHARM_DATASET_WORKERS = 1
PYCHARM_DATASET_VARIANTS = {}
PYCHARM_FRESH = False
PYCHARM_VALIDATE_ONLY = False

MODEL_NAME = os.environ.get("GRAPHREASON_MODEL", "gpt-4o-mini")
SAMPLING_MODE = "balanced"
N_PER_CLASS = 500
TOP_K = 8
NEIGHBOR_RATIO = 0.6
PRKNS_METHOD = "followers"
N_RUNS = 1
BASE_SEED = 2026

USE_WGSE = True
USE_SEHCI = True
USE_PRKNS = True
RUN_ABLATIONS = False
RUN_FEATURE_ABLATIONS = False
FEATURE_MODE = "all"

H = 3
ALPHA = 0.5
BETA = 0.5
PLATEAU_RATIO = 0.02
K_RANGE = range(1, 15)

PRINT_PROMPT = False
VERBOSE = True
SAVE_RUN_DETAILS = False
MAX_CONSECUTIVE_UNKNOWNS = 10

# Provider configuration.  The generic model path uses an OpenAI-compatible
# endpoint; DeepSeek and Zhipu AI can also be selected by their aliases.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_REASONING_EFFORT = os.environ.get("DEEPSEEK_REASONING_EFFORT", "high")
DEEPSEEK_THINKING_ENABLED = os.environ.get(
    "DEEPSEEK_THINKING_ENABLED", "true"
).lower() in {"1", "true", "yes", "on"}

ZHIPUAI_API_KEY = os.environ.get("ZHIPUAI_API_KEY")
ZHIPUAI_MODEL = os.environ.get("ZHIPUAI_MODEL", "GLM-4.7")

PRETRAIN_MODEL_PATH = os.environ.get(
    "PRETRAIN_MODEL_PATH", "google-bert/bert-base-uncased"
)
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_SLEEP = 3
LLM_TIMEOUT = 30
