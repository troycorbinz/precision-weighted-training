"""
ab_compare.py — Blind A/B comparison webapp for two CLLM models.

Two modes:
  1. Interactive: Load models, generate responses live for ad-hoc questions
  2. Batch eval: Serve pre-generated responses from a prepared eval set

Usage (interactive):
    python scripts/ab_compare.py --model-a models/cllm-v1.5-025 --model-b models/cllm-v1.5-026

Usage (pre-generate batch eval responses):
    python scripts/ab_compare.py --generate-batch \\
        --model-a models/cllm-v1.5-025 --model-b models/cllm-v1.5-026 \\
        --questions configs/eval_questions_1.2B.json \\
        --generations 3

Usage (serve batch eval to judges):
    python scripts/ab_compare.py --batch logs/batch_eval.json

Then open http://localhost:8400 in a browser.
"""

import argparse
import json
import random
import hashlib
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from math import comb
from pathlib import Path

# Add project root to sys.path so `from src import ...` works regardless of
# whether the script is launched via `python scripts/ab_compare.py` or
# `python -m scripts.ab_compare` or from a terminal without PYTHONPATH set.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from flask import Flask, render_template_string, request, jsonify, session as flask_session

from src import DEVICE
from src.ModelSession import ModelSession
from src.inference_utils import generate_response

app = Flask(__name__)
app.secret_key = "cllm-ab-compare-2026"


@app.after_request
def no_cache(response):
    """Prevent CDN / browser caching of dynamic endpoints.

    This app has no static content worth caching — every response reflects
    session or state (who the judge is, how many votes have been recorded,
    which pair to show next). Setting no-store on every response avoids
    Cloudflare or browser caches serving stale state to judges.
    """
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Global state
sessions = {}           # 'a' and 'b' ModelSessions (interactive mode)
model_labels = {}       # maps 'a'/'b' to actual model directory names
batch_data = None       # loaded batch eval data (batch mode)
results_file = None     # path to results JSON
demographics_file = None  # path to demographics JSON (per-judge survey answers)
batch_mode = False      # True when serving pre-generated batch


# ─── Response generation ─────────────────────────────────────────────

def load_model(model_dir: str) -> ModelSession:
    """Load a model for inference."""
    print(f"Loading model from {model_dir}...")
    session = ModelSession(model_dir=model_dir, mode="inference")
    session.load_model_components()
    session.load_checkpoint()
    session.get_model().eval()
    print(f"  Loaded: {session.get_model_id()}")
    return session


def generate_from_session(session: ModelSession, prompt: str,
                          temperature: float = 0.7, top_k: int = 40,
                          max_tokens: int = 200) -> str:
    """Generate a response from a model session."""
    vocab = session.get_vocab()
    token_ids, _ = vocab.encode(prompt)
    index = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

    stop_tokens = set()
    if hasattr(vocab, 'end_id') and vocab.end_id is not None:
        stop_tokens.add(vocab.end_id)
    if hasattr(vocab, 'pad_id') and vocab.pad_id is not None:
        stop_tokens.add(vocab.pad_id)

    response, _ = generate_response(
        session=session,
        index_tensor=index,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=0.93,
        stop_tokens=stop_tokens if stop_tokens else None,
        parser="markdown"
    )
    return response


def generate_batch(model_a_dir: str, model_b_dir: str, questions_path: str,
                   n_generations: int = 3, temperature: float = 0.7,
                   top_k: int = 40, max_tokens: int = 200) -> dict:
    """Pre-generate responses from both models for all questions."""
    questions = json.loads(Path(questions_path).read_text())
    session_a = load_model(model_a_dir)
    session_b = load_model(model_b_dir)

    batch = {
        "metadata": {
            "model_a": Path(model_a_dir).name,
            "model_b": Path(model_b_dir).name,
            "questions_file": questions_path,
            "n_generations": n_generations,
            "temperature": temperature,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "generated_at": datetime.now().isoformat(),
        },
        "questions": []
    }

    total = len(questions) * n_generations * 2
    done = 0

    for q in questions:
        q_entry = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "responses_a": [],
            "responses_b": [],
        }

        for gen_idx in range(n_generations):
            print(f"  [{done+1}/{total}] Q{q['id']} gen {gen_idx+1} model A...")
            resp_a = generate_from_session(session_a, q["question"],
                                           temperature, top_k, max_tokens)
            q_entry["responses_a"].append(resp_a)
            done += 1

            print(f"  [{done+1}/{total}] Q{q['id']} gen {gen_idx+1} model B...")
            resp_b = generate_from_session(session_b, q["question"],
                                           temperature, top_k, max_tokens)
            q_entry["responses_b"].append(resp_b)
            done += 1

        batch["questions"].append(q_entry)
        print(f"  Completed question {q['id']}/{len(questions)}")

    return batch


# ─── Pairing logic ───────────────────────────────────────────────────

def get_pairings_for_judge(judge_id: int, questions: list, n_gens: int) -> list:
    """Build the list of (question, response_a_idx, response_b_idx, left_is_a)
    for a specific judge. a_gen, b_gen, and left_is_a are all drawn from a
    per-judge seeded RNG. This avoids the modular-collision failure of a
    linear `(q*k + jid) % n_gens` scheme, where any two judges whose
    judge_id differs by a multiple of n_gens would see identical responses
    on that side for every question."""

    pairings = []

    # Single per-judge RNG drives all three random choices (a_gen, b_gen, left_is_a).
    # Deterministic per judge so a judge resuming a session sees the same sequence.
    judge_rng = random.Random(hashlib.sha256(f"judge-{judge_id}".encode()).hexdigest())

    for q_idx, q in enumerate(questions):
        a_gen = judge_rng.randrange(n_gens)
        b_gen = judge_rng.randrange(n_gens)
        left_is_a = judge_rng.random() < 0.5

        pairings.append({
            "question_idx": q_idx,
            "question_id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "a_gen_idx": a_gen,
            "b_gen_idx": b_gen,
            "left_is_a": left_is_a,
        })

    return pairings


# ─── Results I/O ─────────────────────────────────────────────────────

def load_results() -> list:
    if results_file and results_file.exists():
        return json.loads(results_file.read_text())
    return []


def save_result(result: dict):
    results_file.parent.mkdir(parents=True, exist_ok=True)
    results = load_results()
    results.append(result)
    results_file.write_text(json.dumps(results, indent=2))


def load_demographics() -> dict:
    if demographics_file and demographics_file.exists():
        return json.loads(demographics_file.read_text())
    return {}


def save_demographics(judge_name: str, survey: dict):
    if not demographics_file:
        return
    demographics_file.parent.mkdir(parents=True, exist_ok=True)
    demographics = load_demographics()
    demographics[judge_name] = {
        'timestamp': datetime.now().isoformat(),
        'survey': survey,
    }
    demographics_file.write_text(json.dumps(demographics, indent=2))


# ─── Report helpers ──────────────────────────────────────────────────

_FM_NAME_MARKERS = (
    'chatgpt', 'gpt', 'claude', 'opus', 'sonnet', 'haiku',
    'gemini', 'deepseek', 'minimax', 'grok', 'llama', 'qwen',
    'kimi', 'mistral', 'command-r', 'perplexity', 'muse spark', 'muse',
)


def _classify_judge(name: str) -> str:
    name_lower = name.lower()
    return 'FM' if any(m in name_lower for m in _FM_NAME_MARKERS) else 'Human'


def _two_sided_binomial_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value for k successes in n trials."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    one_sided = sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i)) for i in range(k + 1))
    return min(1.0, 2 * one_sided)


def _compute_judge_stats(name: str, records: list) -> dict:
    records = sorted(records, key=lambda r: r['timestamp'])
    a = sum(1 for r in records if r['winner'] == 'a')
    b = sum(1 for r in records if r['winner'] == 'b')
    t = sum(1 for r in records if r['winner'] == 'tie')
    decisive = a + b
    ts = [datetime.fromisoformat(r['timestamp']) for r in records]
    gaps = [(ts[i+1] - ts[i]).total_seconds() for i in range(len(ts) - 1)]
    return {
        'name': name,
        'type': _classify_judge(name),
        'n': len(records),
        'a': a, 'b': b, 'ties': t,
        'decisive': decisive,
        'b_pct_decisive': (100 * b / decisive) if decisive else None,
        'median_seconds': statistics.median(gaps) if gaps else None,
        'tie_rate': (100 * t / len(records)) if records else 0,
        'first_ts': records[0]['timestamp'][:19].replace('T', ' ') if records else '',
        'last_ts': records[-1]['timestamp'][:19].replace('T', ' ') if records else '',
    }


def _compute_aggregate(records: list) -> dict:
    a = sum(1 for r in records if r['winner'] == 'a')
    b = sum(1 for r in records if r['winner'] == 'b')
    t = sum(1 for r in records if r['winner'] == 'tie')
    decisive = a + b
    return {
        'total': len(records),
        'a': a, 'b': b, 'ties': t,
        'decisive': decisive,
        'a_pct_decisive': (100 * a / decisive) if decisive else None,
        'b_pct_decisive': (100 * b / decisive) if decisive else None,
        'tie_pct': (100 * t / len(records)) if records else None,
        'p_value': _two_sided_binomial_p(min(a, b), decisive) if decisive else None,
    }


# ─── HTML Template ───────────────────────────────────────────────────

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>CLLM Blind Comparison</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: #161618; color: #e0e0e0; padding: 20px; }
        h1 { text-align: center; color: #7c83ff; margin-bottom: 5px; }
        .subtitle { text-align: center; color: #666; margin-bottom: 20px; font-size: 13px; }
        .stats { text-align: center; color: #888; margin-bottom: 20px; font-size: 14px; }

        /* Login */
        .login-area { max-width: 400px; margin: 60px auto; text-align: center; }
        .login-area input {
            padding: 12px; border: 1px solid #2a2a30; border-radius: 8px;
            background: #1e1e22; color: #e0e0e0; font-size: 16px; width: 100%;
            margin-bottom: 10px;
        }
        .login-area button {
            padding: 10px 30px; background: #7c83ff; color: white;
            border: none; border-radius: 6px; font-size: 16px; cursor: pointer;
        }

        /* Interactive input */
        .input-area { max-width: 800px; margin: 0 auto 20px; }
        .input-area textarea {
            width: 100%; padding: 12px; border: 1px solid #2a2a30; border-radius: 8px;
            background: #1e1e22; color: #e0e0e0; font-size: 16px; resize: vertical;
            min-height: 80px;
        }
        .input-area button {
            margin-top: 10px; padding: 10px 30px; background: #7c83ff; color: white;
            border: none; border-radius: 6px; font-size: 16px; cursor: pointer;
        }
        .input-area button:hover { background: #6a70e0; }
        .input-area button:disabled { background: #444; cursor: not-allowed; }

        /* Batch prompt display */
        .prompt-display {
            max-width: 800px; margin: 0 auto 20px; padding: 15px 18px 15px 20px;
            background: #1a2332; border-left: 4px solid #7c83ff;
            border-radius: 4px;
            font-size: 16px; line-height: 1.5;
        }
        .prompt-label { color: #7c83ff; font-size: 12px; text-transform: uppercase; margin-bottom: 5px; }
        .progress-bar {
            max-width: 800px; margin: 0 auto 15px; background: #1e1e22;
            border-radius: 8px; height: 8px; overflow: hidden;
        }
        .progress-fill { height: 100%; background: #7c83ff; transition: width 0.3s; }
        .progress-text { max-width: 800px; margin: 0 auto 15px; text-align: center; color: #888; font-size: 13px; }

        /* Responses */
        .responses { display: flex; gap: 20px; max-width: 1400px; margin: 0 auto; }
        .response-box {
            flex: 1; background: #1e1e22; border: 1px solid #2a2a30; border-radius: 8px;
            padding: 20px; min-height: 200px;
        }
        .response-box h2 { color: #7c83ff; margin-bottom: 10px; font-size: 18px; }
        .response-text { white-space: pre-wrap; line-height: 1.6; font-size: 15px; min-height: 100px; }
        .vote-btn {
            margin-top: 15px; padding: 8px 20px; border: 2px solid #7c83ff; background: transparent;
            color: #7c83ff; border-radius: 6px; cursor: pointer; font-size: 14px; width: 100%;
        }
        .vote-btn:hover { background: #7c83ff; color: white; }
        .vote-btn:disabled { border-color: #444; color: #444; cursor: not-allowed; background: transparent; }
        .tie-btn { margin-top: 15px; text-align: center; }
        .tie-btn button {
            padding: 8px 20px; border: 2px solid #666; background: transparent;
            color: #888; border-radius: 6px; cursor: pointer; font-size: 14px;
        }
        .tie-btn button:hover { border-color: #aaa; color: #aaa; }

        .reveal { text-align: center; margin-top: 20px; padding: 15px; background: #1e1e22;
            border-radius: 8px; max-width: 800px; margin-left: auto; margin-right: auto; }
        .reveal.winner-a { border: 2px solid #4caf50; }
        .reveal.winner-b { border: 2px solid #ff9800; }

        .spinner { display: none; text-align: center; padding: 40px; font-size: 18px; color: #7c83ff; }

        .complete { text-align: center; padding: 60px; font-size: 22px; color: #4caf50; }

        /* Guide modal */
        .modal-overlay {
            display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.7); z-index: 100; justify-content: center; align-items: center;
        }
        .modal-overlay.active { display: flex; }
        .modal {
            background: #1e1e22; border: 1px solid #7c83ff; border-radius: 12px;
            padding: 30px; max-width: 600px; width: 90%; max-height: 80vh; overflow-y: auto;
        }
        .modal h2 { color: #7c83ff; margin-bottom: 15px; text-align: center; }
        .modal h3 { color: #7c83ff; margin: 15px 0 8px; font-size: 15px; }
        .modal p, .modal li { line-height: 1.6; font-size: 14px; color: #ccc; }
        .modal ul { margin-left: 20px; margin-bottom: 10px; }
        .modal .criteria-item { margin-bottom: 8px; }
        .modal .criteria-label { color: #7c83ff; font-weight: bold; }
        .modal .note { margin-top: 15px; padding: 10px; background: #1a2332; border-radius: 6px; font-size: 13px; color: #aaa; }
        .modal button {
            margin-top: 20px; padding: 10px 30px; background: #7c83ff; color: white;
            border: none; border-radius: 6px; font-size: 16px; cursor: pointer; width: 100%;
        }
        .modal button:hover { background: #6a70e0; }

        .copy-btn {
            display: block; max-width: 800px; margin: 0 auto 15px; padding: 8px 16px;
            background: transparent; border: 1px solid #555; color: #888;
            border-radius: 6px; cursor: pointer; font-size: 13px; text-align: center;
        }
        .copy-btn:hover { border-color: #7c83ff; color: #7c83ff; }
        .copy-btn.copied { border-color: #4caf50; color: #4caf50; }

        /* Demographic survey */
        #survey-area { max-width: 600px; margin: 40px auto; }
        #survey-area h2 { color: #7c83ff; margin-bottom: 10px; text-align: center; }
        #survey-area .survey-intro { color: #aaa; margin-bottom: 25px; font-size: 13px; text-align: center; }
        .survey-q { margin-bottom: 18px; }
        .survey-q label { display: block; margin-bottom: 6px; color: #e0e0e0; font-size: 14px; }
        .survey-q select, .survey-q input[type="text"] {
            width: 100%; padding: 10px; border: 1px solid #2a2a30; border-radius: 6px;
            background: #1e1e22; color: #e0e0e0; font-size: 15px;
        }
        #survey-area button {
            padding: 10px 30px; background: #7c83ff; color: white;
            border: none; border-radius: 6px; font-size: 16px; cursor: pointer;
            margin-top: 10px; width: 100%;
        }
        #survey-area button:hover { background: #6a70e0; }
        #thanks-area { text-align: center; padding: 60px 20px; max-width: 600px; margin: 0 auto; }
        #thanks-area h2 { color: #4caf50; margin-bottom: 15px; }
        #thanks-area p { color: #aaa; font-size: 15px; line-height: 1.5; }
    </style>
</head>
<body>
    <h1>CLLM Blind A/B Comparison</h1>
    <div class="subtitle" id="mode-label"></div>
    <div class="stats" id="stats"></div>

    <!-- Login screen (batch mode) -->
    <div class="login-area" id="login-area" style="display:none;">
        <h2 style="color:#7c83ff; margin-bottom:20px;">Enter your name to begin</h2>
        <input type="text" id="judge-name" placeholder="Your name...">
        <br>
        <button onclick="login()">Continue</button>
    </div>

    <!-- Demographic survey (shown after name entry) -->
    <div id="survey-area" style="display:none;">
        <h2>A few quick questions about you</h2>
        <p class="survey-intro">
            This is for demographic context in the paper. Your individual answers aren't published — only aggregate summaries (e.g., "3 of N judges were daily LLM users"). Takes under a minute. Leave any question blank if you prefer.
        </p>

        <div class="survey-q">
            <label>How often do you use LLM chatbots (ChatGPT, Claude, Gemini, etc.)?</label>
            <select id="s-frequency">
                <option value="">— select —</option>
                <option value="never">Never or almost never</option>
                <option value="occasional">Occasionally (a few times a month)</option>
                <option value="weekly">Weekly</option>
                <option value="daily">Daily or near-daily</option>
            </select>
        </div>

        <div class="survey-q">
            <label>Technical background?</label>
            <select id="s-background">
                <option value="">— select —</option>
                <option value="non-technical">Non-technical</option>
                <option value="technical-non-ml">Technical (engineering, CS, etc.) but not ML-focused</option>
                <option value="ml-or-cs-research">ML or CS research</option>
            </select>
        </div>

        <div class="survey-q">
            <label>Primary language you read/write in?</label>
            <input type="text" id="s-language" placeholder="e.g., English">
        </div>

        <div class="survey-q">
            <label>Age range?</label>
            <select id="s-age">
                <option value="">— select —</option>
                <option value="under-25">Under 25</option>
                <option value="25-40">25–40</option>
                <option value="40-60">40–60</option>
                <option value="over-60">Over 60</option>
            </select>
        </div>

        <div class="survey-q">
            <label>Have you participated in model evaluation or A/B preference studies before?</label>
            <select id="s-prior">
                <option value="">— select —</option>
                <option value="yes">Yes</option>
                <option value="no">No</option>
            </select>
        </div>

        <button onclick="submitSurvey()">Submit</button>
    </div>

    <!-- Thank-you screen (followup mode only) -->
    <div id="thanks-area" style="display:none;">
        <h2>Thank you!</h2>
        <p>Your responses have been recorded. That's all I needed — no need to go through the judging interface.</p>
    </div>

    <!-- Batch progress -->
    <div id="batch-progress" style="display:none;">
        <div class="progress-text" id="progress-text"></div>
        <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
    </div>

    <!-- Batch prompt display -->
    <div class="prompt-display" id="prompt-display" style="display:none;">
        <div class="prompt-label">Question <span id="q-number"></span> (<span id="q-category"></span>)</div>
        <div id="prompt-text"></div>
    </div>

    <!-- Interactive input -->
    <div class="input-area" id="interactive-area" style="display:none;">
        <textarea id="prompt" placeholder="Ask a question..."></textarea>
        <button id="generate-btn" onclick="generateInteractive()">Generate</button>
    </div>

    <div class="spinner" id="spinner">Generating responses...</div>

    <div class="responses" id="responses" style="display:none;">
        <div class="response-box">
            <h2>Response Left</h2>
            <div class="response-text" id="response-left"></div>
            <button class="vote-btn" id="vote-left" onclick="vote('left')">This one is better</button>
        </div>
        <div class="response-box">
            <h2>Response Right</h2>
            <div class="response-text" id="response-right"></div>
            <button class="vote-btn" id="vote-right" onclick="vote('right')">This one is better</button>
        </div>
    </div>
    <div class="tie-btn" id="tie-area" style="display:none;">
        <button onclick="vote('tie')">Both are equal / Can't decide</button>
    </div>
    <button class="copy-btn" id="copy-btn" style="display:none;" onclick="copyToClipboard()">Copy question + responses to clipboard</button>

    <div class="reveal" id="reveal" style="display:none;"></div>
    <div class="complete" id="complete" style="display:none;">
        Thank you! You've completed all evaluations.
    </div>

    <!-- Guide modal -->
    <div class="modal-overlay" id="guide-modal">
        <div class="modal">
            <h2>How to Judge These Responses</h2>
            <p>You'll be comparing responses from two AI models that are still early in training.
            They will <strong>NOT</strong> produce correct or complete answers &mdash; that's expected and normal.</p>

            <h3>Judge based on:</h3>
            <ul>
                <li class="criteria-item"><span class="criteria-label">Relevance</span> &mdash; Did the response attempt to address the question? Even partially? Mentioning a key word or concept from the prompt counts.</li>
                <li class="criteria-item"><span class="criteria-label">Coherence</span> &mdash; Does the text flow naturally? Are sentences grammatically structured, even if the content is wrong?</li>
                <li class="criteria-item"><span class="criteria-label">Diversity</span> &mdash; Does the response show varied vocabulary and structure, or does it repeat the same phrase over and over?</li>
                <li class="criteria-item"><span class="criteria-label">Human-likeness</span> &mdash; Does it read like something a person might write (even if confused)? Or does it look like a database dump or raw code?</li>
            </ul>

            <div class="note">
                When in doubt, go with your gut. If both responses seem equally bad or equally good, pick "Both are equal / Can't decide." There are 32 questions and it takes about 10-15 minutes.
            </div>

            <button onclick="dismissGuide()">Got it &mdash; let's start!</button>
        </div>
    </div>

    <script>
        const BATCH_MODE = {{ batch_mode | tojson }};
        const FOLLOWUP_MODE = {{ followup_mode | tojson }};
        let judgeName = '';
        let pairings = [];
        let currentIdx = 0;

        function updateStats() {
            fetch('/stats').then(r => r.json()).then(data => {
                if (BATCH_MODE) {
                    document.getElementById('stats').innerHTML =
                        `Total votes: ${data.total} | Model A wins: ${data.a_wins} | Model B wins: ${data.b_wins} | Ties: ${data.ties} | Judges: ${data.judge_count}`;
                } else {
                    document.getElementById('stats').innerHTML =
                        `Comparisons: ${data.total} | Model A wins: ${data.a_wins} | Model B wins: ${data.b_wins} | Ties: ${data.ties}`;
                }
            });
        }

        function init() {
            updateStats();
            if (FOLLOWUP_MODE) {
                document.getElementById('mode-label').textContent = 'Follow-up Survey';
                document.getElementById('login-area').style.display = 'block';
            } else if (BATCH_MODE) {
                document.getElementById('mode-label').textContent = 'Batch Evaluation Mode';
                document.getElementById('login-area').style.display = 'block';
            } else {
                document.getElementById('mode-label').textContent = 'Interactive Mode';
                document.getElementById('interactive-area').style.display = 'block';
            }
        }

        function login() {
            judgeName = document.getElementById('judge-name').value.trim();
            if (!judgeName) return;
            document.getElementById('login-area').style.display = 'none';
            // Always go through survey first (followup or batch)
            document.getElementById('survey-area').style.display = 'block';
        }

        function submitSurvey() {
            const survey = {
                frequency: document.getElementById('s-frequency').value,
                background: document.getElementById('s-background').value,
                language: document.getElementById('s-language').value.trim(),
                age: document.getElementById('s-age').value,
                prior_studies: document.getElementById('s-prior').value,
            };
            fetch('/survey', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ judge_name: judgeName, survey: survey })
            }).then(r => r.json()).then(() => {
                document.getElementById('survey-area').style.display = 'none';
                if (FOLLOWUP_MODE) {
                    document.getElementById('thanks-area').style.display = 'block';
                } else {
                    startJudging();
                }
            });
        }

        function startJudging() {
            // Fetch pairings first, then show guide (original login() behavior)
            fetch('/batch/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ judge_name: judgeName })
            }).then(r => r.json()).then(data => {
                pairings = data.pairings;
                currentIdx = data.resume_from;
                if (currentIdx >= pairings.length) {
                    document.getElementById('complete').style.display = 'block';
                } else if (currentIdx === 0) {
                    // First time — show the guide
                    document.getElementById('guide-modal').classList.add('active');
                } else {
                    // Resuming — skip the guide
                    showBatchQuestion();
                }
            });
        }

        function dismissGuide() {
            document.getElementById('guide-modal').classList.remove('active');
            showBatchQuestion();
        }

        function showBatchQuestion() {
            if (currentIdx >= pairings.length) {
                document.getElementById('responses').style.display = 'none';
                document.getElementById('tie-area').style.display = 'none';
                document.getElementById('prompt-display').style.display = 'none';
                document.getElementById('reveal').style.display = 'none';
                document.getElementById('complete').style.display = 'block';
                updateStats();
                return;
            }

            let p = pairings[currentIdx];
            document.getElementById('batch-progress').style.display = 'block';
            document.getElementById('progress-text').textContent =
                `Question ${currentIdx + 1} of ${pairings.length}`;
            document.getElementById('progress-fill').style.width =
                `${(currentIdx / pairings.length) * 100}%`;

            document.getElementById('prompt-display').style.display = 'block';
            document.getElementById('q-number').textContent = p.question_id;
            document.getElementById('q-category').textContent = p.category;
            document.getElementById('prompt-text').textContent = p.question;

            // Fetch the pre-generated responses
            fetch('/batch/pair', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    question_idx: p.question_idx,
                    a_gen_idx: p.a_gen_idx,
                    b_gen_idx: p.b_gen_idx,
                    left_is_a: p.left_is_a
                })
            }).then(r => r.json()).then(data => {
                document.getElementById('response-left').textContent = data.left;
                document.getElementById('response-right').textContent = data.right;
                document.getElementById('responses').style.display = 'flex';
                document.getElementById('tie-area').style.display = 'block';
                document.getElementById('copy-btn').style.display = 'block';
                document.getElementById('reveal').style.display = 'none';
                document.getElementById('vote-left').disabled = false;
                document.getElementById('vote-right').disabled = false;
            });
        }

        function vote(choice) {
            document.getElementById('vote-left').disabled = true;
            document.getElementById('vote-right').disabled = true;

            let body;
            if (BATCH_MODE) {
                let p = pairings[currentIdx];
                body = {
                    choice: choice,
                    judge_name: judgeName,
                    question_id: p.question_id,
                    question_idx: p.question_idx,
                    a_gen_idx: p.a_gen_idx,
                    b_gen_idx: p.b_gen_idx,
                    left_is_a: p.left_is_a
                };
            } else {
                body = { choice: choice, prompt: document.getElementById('prompt').value.trim() };
            }

            fetch('/vote', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(body)
            }).then(r => r.json()).then(data => {
                let reveal = document.getElementById('reveal');
                reveal.className = 'reveal';
                if (data.winner === 'tie') {
                    reveal.innerHTML = `<strong>Tie!</strong><br>Left was: <strong>${data.left_label}</strong> | Right was: <strong>${data.right_label}</strong>`;
                } else {
                    reveal.innerHTML = `<strong>You picked: ${data.winner_side}</strong> (<strong>${data.winner_label}</strong>)<br>` +
                        `Left was: <strong>${data.left_label}</strong> | Right was: <strong>${data.right_label}</strong>`;
                    reveal.classList.add(data.winner_id === 'a' ? 'winner-a' : 'winner-b');
                }
                reveal.style.display = 'block';
                document.getElementById('tie-area').style.display = 'none';
                updateStats();

                document.getElementById('copy-btn').style.display = 'none';
                if (BATCH_MODE) {
                    setTimeout(() => {
                        currentIdx++;
                        showBatchQuestion();
                    }, 2000);
                }
            });
        }

        // Interactive mode
        function generateInteractive() {
            let prompt = document.getElementById('prompt').value.trim();
            if (!prompt) return;

            document.getElementById('generate-btn').disabled = true;
            document.getElementById('responses').style.display = 'none';
            document.getElementById('reveal').style.display = 'none';
            document.getElementById('tie-area').style.display = 'none';
            document.getElementById('spinner').style.display = 'block';

            fetch('/generate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ prompt: prompt })
            }).then(r => r.json()).then(data => {
                document.getElementById('response-left').textContent = data.left;
                document.getElementById('response-right').textContent = data.right;
                document.getElementById('spinner').style.display = 'none';
                document.getElementById('responses').style.display = 'flex';
                document.getElementById('tie-area').style.display = 'block';
                document.getElementById('vote-left').disabled = false;
                document.getElementById('vote-right').disabled = false;
                document.getElementById('generate-btn').disabled = false;
            }).catch(err => {
                document.getElementById('spinner').textContent = 'Error: ' + err;
                document.getElementById('generate-btn').disabled = false;
            });
        }

        function copyToClipboard() {
            let question = '';
            if (BATCH_MODE && currentIdx < pairings.length) {
                question = pairings[currentIdx].question;
            } else {
                question = document.getElementById('prompt')?.value || '';
            }
            let left = document.getElementById('response-left').textContent;
            let right = document.getElementById('response-right').textContent;

            let text = `Question: ${question}\n\n` +
                `--- Response A ---\n${left}\n\n` +
                `--- Response B ---\n${right}\n\n` +
                `Which response is better? Judge based on: relevance to the question, coherence and grammar, vocabulary diversity, and human-likeness. These are early-training AI models so neither answer will be correct — judge on quality of attempt. Answer with just "A", "B", or "Tie".`;

            navigator.clipboard.writeText(text).then(() => {
                let btn = document.getElementById('copy-btn');
                btn.textContent = 'Copied!';
                btn.classList.add('copied');
                setTimeout(() => {
                    btn.textContent = 'Copy question + responses to clipboard';
                    btn.classList.remove('copied');
                }, 2000);
            });
        }

        document.getElementById('prompt')?.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); generateInteractive(); }
        });

        init();
    </script>
</body>
</html>
"""


REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<title>A/B Judging Report</title>
<meta http-equiv="refresh" content="30">
<style>
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #161618; color: #e0e0e0; padding: 20px; max-width: 1300px; margin: 0 auto; }
  h1 { color: #7c83ff; margin-bottom: 4px; }
  .updated { color: #888; font-size: 12px; margin-bottom: 24px; }
  .section { background: #1e1e22; border: 1px solid #2a2a30; border-radius: 6px; padding: 16px 20px; margin-bottom: 16px; }
  .section h2 { color: #7c83ff; margin-bottom: 12px; font-size: 16px; }
  .kv { display: flex; gap: 28px; flex-wrap: wrap; }
  .kv-item { font-size: 14px; }
  .kv-item .k { color: #888; }
  .kv-item .v { color: #e0e0e0; font-weight: 600; }
  .highlight { color: #7c83ff; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { background: #1a2332; color: #7c83ff; padding: 8px; text-align: left; border-bottom: 1px solid #2a2a30; font-weight: 600; }
  td { padding: 6px 8px; border-bottom: 1px solid #2a2a30; }
  tr:hover td { background: #1a2332; }
  .type-FM { color: #6ea8b8; }
  .type-Human { color: #e0e0e0; }
  .numeric { text-align: right; font-variant-numeric: tabular-nums; }
  .warn { color: #ff9800; }
  .demo-bar { color: #ccc; font-size: 13px; margin: 5px 0; }
  .demo-bar .label { color: #888; display: inline-block; min-width: 130px; }
</style>
</head>
<body>
  <h1>A/B Judging Report</h1>
  <div class="updated">Updated: {{ ts_now }} &middot; auto-refreshes every 30 seconds</div>

  <div class="section">
    <h2>Aggregate</h2>
    <div class="kv">
      <div class="kv-item"><span class="k">Total judgments:</span> <span class="v">{{ agg.total }}</span></div>
      <div class="kv-item"><span class="k">Judges:</span> <span class="v">{{ n_judges }}</span></div>
      <div class="kv-item"><span class="k">Decisive (A+B):</span> <span class="v">{{ agg.decisive }}</span></div>
      <div class="kv-item"><span class="k">A wins:</span> <span class="v">{{ agg.a }}</span></div>
      <div class="kv-item"><span class="k">B wins:</span> <span class="v">{{ agg.b }}</span></div>
      <div class="kv-item"><span class="k">Ties:</span> <span class="v">{{ agg.ties }}</span></div>
      {% if agg.b_pct_decisive is not none %}
      <div class="kv-item"><span class="k">B% of decisive:</span> <span class="v highlight">{{ "%.1f"|format(agg.b_pct_decisive) }}%</span></div>
      <div class="kv-item"><span class="k">Two-sided binomial p:</span> <span class="v highlight">{{ "%.2e"|format(agg.p_value) }}</span></div>
      {% endif %}
    </div>
  </div>

  {% if sensitivity %}
  <div class="section">
    <h2>Sensitivity</h2>
    <table>
      <tr><th>Filter</th><th class="numeric">Judges</th><th class="numeric">Decisive</th><th class="numeric">B%</th><th class="numeric">p-value</th></tr>
      {% for s in sensitivity %}
      <tr>
        <td>{{ s.label }}</td>
        <td class="numeric">{{ s.n_judges }}</td>
        <td class="numeric">{{ s.decisive }}</td>
        <td class="numeric">{% if s.b_pct_decisive is not none %}{{ "%.1f"|format(s.b_pct_decisive) }}%{% else %}&mdash;{% endif %}</td>
        <td class="numeric">{% if s.p_value is not none %}{{ "%.2e"|format(s.p_value) }}{% else %}&mdash;{% endif %}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
  {% endif %}

  <div class="section">
    <h2>Per-judge</h2>
    <table>
      <tr>
        <th>Judge</th><th>Type</th>
        <th class="numeric">N</th><th class="numeric">A</th><th class="numeric">B</th><th class="numeric">Tie</th>
        <th class="numeric">Tie%</th><th class="numeric">B% dec</th><th class="numeric">Median time</th>
        <th>First</th><th>Last</th>
      </tr>
      {% for j in judges %}
      <tr>
        <td>{{ j.name }}</td>
        <td class="type-{{ j.type }}">{{ j.type }}</td>
        <td class="numeric">{{ j.n }}</td>
        <td class="numeric">{{ j.a }}</td>
        <td class="numeric">{{ j.b }}</td>
        <td class="numeric">{{ j.ties }}</td>
        <td class="numeric {% if j.tie_rate > 80 %}warn{% endif %}">{{ "%.0f"|format(j.tie_rate) }}%</td>
        <td class="numeric">{% if j.b_pct_decisive is not none %}{{ "%.1f"|format(j.b_pct_decisive) }}%{% else %}&mdash;{% endif %}</td>
        <td class="numeric {% if j.median_seconds is not none and j.median_seconds < 15 %}warn{% endif %}">{% if j.median_seconds is not none %}{{ "%.1f"|format(j.median_seconds) }}s{% else %}&mdash;{% endif %}</td>
        <td>{{ j.first_ts }}</td>
        <td>{{ j.last_ts }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>

  <div class="section">
    <h2>Demographics ({{ demo.n }} surveyed)</h2>
    {% if demo.n == 0 %}
    <div class="demo-bar">No demographics collected yet.</div>
    {% else %}
    {% for cat, items in demo.breakdown.items() %}
    <div class="demo-bar">
      <span class="label">{{ cat }}:</span>
      {% for key, count in items %}{{ key or "(blank)" }} ({{ count }}){% if not loop.last %} &middot; {% endif %}{% endfor %}
    </div>
    {% endfor %}
    {% endif %}
  </div>

  <div class="section">
    <h2>Per-question coverage ({{ coverage_summary.n_questions }} questions, min/max votes per question: {{ coverage_summary.min }} / {{ coverage_summary.max }})</h2>
    <table>
      <tr>
        <th class="numeric">Q#</th>
        <th>Prompt (snippet)</th>
        <th class="numeric">Votes</th>
        <th class="numeric">A</th>
        <th class="numeric">B</th>
        <th class="numeric">Tie</th>
        <th class="numeric">B% dec</th>
      </tr>
      {% for c in coverage %}
      <tr>
        <td class="numeric">{{ c.qid }}</td>
        <td>{{ c.snippet }}</td>
        <td class="numeric {% if c.n < coverage_summary.max %}warn{% endif %}">{{ c.n }}</td>
        <td class="numeric">{{ c.a }}</td>
        <td class="numeric">{{ c.b }}</td>
        <td class="numeric">{{ c.ties }}</td>
        <td class="numeric">{% if c.b_pct_decisive is not none %}{{ "%.0f"|format(c.b_pct_decisive) }}%{% else %}&mdash;{% endif %}</td>
      </tr>
      {% endfor %}
    </table>
  </div>

  <div class="section">
    <h2>Per-answer coverage ({{ answer_summary.n_answers }} unique answers, min/max shown: {{ answer_summary.min }} / {{ answer_summary.max }}{% if answer_summary.never_shown %}, <span class="warn">{{ answer_summary.never_shown }} never shown</span>{% endif %})</h2>
    <table>
      <tr>
        <th class="numeric">Q#</th>
        <th>Model</th>
        <th class="numeric">Gen</th>
        <th class="numeric">Shown</th>
        <th class="numeric">Won</th>
        <th class="numeric">Tied</th>
        <th class="numeric">Win%</th>
      </tr>
      {% for a in answers %}
      <tr>
        <td class="numeric">{{ a.qid }}</td>
        <td class="type-{% if a.model == 'A' %}Human{% else %}FM{% endif %}">{{ a.model }}</td>
        <td class="numeric">{{ a.gen }}</td>
        <td class="numeric {% if a.shown == 0 %}warn{% endif %}">{{ a.shown }}</td>
        <td class="numeric">{{ a.won }}</td>
        <td class="numeric">{{ a.tied }}</td>
        <td class="numeric">{% if a.win_pct is not none %}{{ "%.0f"|format(a.win_pct) }}%{% else %}&mdash;{% endif %}</td>
      </tr>
      {% endfor %}
    </table>
  </div>

  <div class="section">
    <h2>Recent activity (last 15 votes)</h2>
    <table>
      <tr><th>When</th><th>Judge</th><th class="numeric">Q#</th><th>Result</th></tr>
      {% for r in recent %}
      <tr>
        <td>{{ r.ts }}</td>
        <td>{{ r.judge }}</td>
        <td class="numeric">{{ r.q_id }}</td>
        <td>{{ r.winner }}</td>
      </tr>
      {% endfor %}
    </table>
  </div>
</body>
</html>
"""


# ─── Flask Routes ────────────────────────────────────────────────────

@app.route('/')
def index():
    followup = request.args.get('followup') == '1'
    return render_template_string(HTML_TEMPLATE, batch_mode=batch_mode, followup_mode=followup)


@app.route('/_report')
def admin_report():
    results = load_results()
    demographics = load_demographics()

    by_judge = defaultdict(list)
    for r in results:
        by_judge[r.get('judge_name', '(unknown)')].append(r)
    judges = [_compute_judge_stats(name, recs) for name, recs in by_judge.items()]
    judges.sort(key=lambda j: j['first_ts'])

    agg = _compute_aggregate(results)

    sensitivity = []
    fm_names = {j['name'] for j in judges if j['type'] == 'FM'}
    human_names = {j['name'] for j in judges if j['type'] == 'Human'}
    fast_names = {j['name'] for j in judges
                  if j['type'] == 'Human'
                  and j.get('median_seconds') is not None
                  and j['median_seconds'] < 15}
    tie_biased = {j['name'] for j in judges if j['tie_rate'] > 80}

    if fm_names:
        filt = [r for r in results if r.get('judge_name') in fm_names]
        a = _compute_aggregate(filt)
        sensitivity.append({
            'label': 'FMs only (' + ', '.join(sorted(fm_names)) + ')',
            'n_judges': len(fm_names),
            **a,
        })
    if human_names:
        filt = [r for r in results if r.get('judge_name') in human_names]
        a = _compute_aggregate(filt)
        sensitivity.append({
            'label': 'Humans only',
            'n_judges': len(human_names),
            **a,
        })

    if fast_names:
        filt = [r for r in results if r.get('judge_name') not in fast_names]
        a = _compute_aggregate(filt)
        sensitivity.append({
            'label': 'Exclude human speed-clickers (median <15s; FMs exempt — fast by nature): ' + ', '.join(sorted(fast_names)),
            'n_judges': len(judges) - len(fast_names),
            **a,
        })
    if tie_biased:
        filt = [r for r in results if r.get('judge_name') not in tie_biased]
        a = _compute_aggregate(filt)
        sensitivity.append({
            'label': 'Exclude tie-biased (>80% ties): ' + ', '.join(sorted(tie_biased)),
            'n_judges': len(judges) - len(tie_biased),
            **a,
        })
    partials = {j['name'] for j in judges if j['n'] < 32}
    if partials:
        filt = [r for r in results if r.get('judge_name') not in partials]
        a = _compute_aggregate(filt)
        sensitivity.append({
            'label': 'Exclude partial completions (n<32, fixed-order prompt-coverage bias): ' + ', '.join(sorted(partials)),
            'n_judges': len(judges) - len(partials),
            **a,
        })
    excl_all = fast_names | tie_biased | partials
    if excl_all and excl_all not in (fast_names, tie_biased, partials):
        filt = [r for r in results if r.get('judge_name') not in excl_all]
        a = _compute_aggregate(filt)
        sensitivity.append({
            'label': 'Exclude all of the above',
            'n_judges': len(judges) - len(excl_all),
            **a,
        })

    demo_breakdown = {}
    survey_items = [v.get('survey', {}) for v in demographics.values()]
    for field in ['frequency', 'background', 'language', 'age', 'prior_studies']:
        counts = Counter((s.get(field) or '').strip() for s in survey_items)
        ordered = sorted([(k, n) for k, n in counts.items()], key=lambda x: -x[1])
        demo_breakdown[field] = ordered

    per_q = defaultdict(lambda: {'a': 0, 'b': 0, 'tie': 0})
    for r in results:
        qid = r.get('question_id', '?')
        w = r['winner']
        if w in ('a', 'b', 'tie'):
            per_q[qid][w] += 1
    q_snippets = {}
    if batch_data:
        for q in batch_data.get('questions', []):
            q_snippets[q['id']] = (q.get('question') or '')[:70]
    coverage = []
    for qid in sorted(per_q.keys(), key=lambda x: (isinstance(x, str), x)):
        counts = per_q[qid]
        n = counts['a'] + counts['b'] + counts['tie']
        dec = counts['a'] + counts['b']
        coverage.append({
            'qid': qid,
            'snippet': q_snippets.get(qid, ''),
            'n': n,
            'a': counts['a'],
            'b': counts['b'],
            'ties': counts['tie'],
            'b_pct_decisive': (100 * counts['b'] / dec) if dec else None,
        })
    cov_counts = [c['n'] for c in coverage]
    coverage_summary = {
        'min': min(cov_counts) if cov_counts else 0,
        'max': max(cov_counts) if cov_counts else 0,
        'n_questions': len(coverage),
    }

    # Per-answer coverage: one row per (question_id, model, gen_idx).
    # Each judgment exposes exactly one A-answer and one B-answer.
    ans_counts = defaultdict(lambda: {'shown': 0, 'won': 0, 'tied': 0})
    for r in results:
        qid = r.get('question_id', '?')
        a_gen = r.get('a_gen_idx')
        b_gen = r.get('b_gen_idx')
        w = r['winner']
        if a_gen is not None:
            key = (qid, 'A', a_gen)
            ans_counts[key]['shown'] += 1
            if w == 'a':
                ans_counts[key]['won'] += 1
            elif w == 'tie':
                ans_counts[key]['tied'] += 1
        if b_gen is not None:
            key = (qid, 'B', b_gen)
            ans_counts[key]['shown'] += 1
            if w == 'b':
                ans_counts[key]['won'] += 1
            elif w == 'tie':
                ans_counts[key]['tied'] += 1
    answers = []
    for (qid, model, gen), c in sorted(ans_counts.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        shown = c['shown']
        answers.append({
            'qid': qid,
            'model': model,
            'gen': gen,
            'shown': shown,
            'won': c['won'],
            'tied': c['tied'],
            'win_pct': (100 * c['won'] / shown) if shown else None,
        })
    ans_shown = [a['shown'] for a in answers]
    answer_summary = {
        'n_answers': len(answers),
        'min': min(ans_shown) if ans_shown else 0,
        'max': max(ans_shown) if ans_shown else 0,
        'never_shown': 0,
    }
    if batch_data:
        n_gens = batch_data['metadata'].get('n_generations', 0)
        expected_keys = set()
        for q in batch_data.get('questions', []):
            for g in range(n_gens):
                expected_keys.add((q['id'], 'A', g))
                expected_keys.add((q['id'], 'B', g))
        actual_keys = set(ans_counts.keys())
        never = expected_keys - actual_keys
        for (qid, model, gen) in sorted(never, key=lambda x: (x[0], x[1], x[2])):
            answers.append({
                'qid': qid, 'model': model, 'gen': gen,
                'shown': 0, 'won': 0, 'tied': 0, 'win_pct': None,
            })
        answer_summary['never_shown'] = len(never)
        answer_summary['n_answers'] = len(expected_keys)
        if never:
            answer_summary['min'] = 0
        answers.sort(key=lambda a: (a['qid'], a['model'], a['gen']))

    sorted_results = sorted(results, key=lambda r: r['timestamp'], reverse=True)
    winner_label = {'a': 'A (baseline)', 'b': 'B (gain)', 'tie': 'Tie'}
    recent = [{
        'ts': r['timestamp'][:19].replace('T', ' '),
        'judge': r.get('judge_name', '?'),
        'q_id': r.get('question_id', '?'),
        'winner': winner_label.get(r['winner'], r['winner']),
    } for r in sorted_results[:15]]

    return render_template_string(
        REPORT_TEMPLATE,
        ts_now=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        agg=agg,
        n_judges=len(judges),
        judges=judges,
        sensitivity=sensitivity,
        demo={'n': len(demographics), 'breakdown': demo_breakdown},
        coverage=coverage,
        coverage_summary=coverage_summary,
        answers=answers,
        answer_summary=answer_summary,
        recent=recent,
    )


@app.route('/followup')
def followup_page():
    return render_template_string(HTML_TEMPLATE, batch_mode=batch_mode, followup_mode=True)


@app.route('/survey', methods=['POST'])
def survey():
    data = request.json or {}
    judge_name = (data.get('judge_name') or '').strip()
    if not judge_name:
        return jsonify({'ok': False, 'error': 'missing judge_name'}), 400
    save_demographics(judge_name, data.get('survey', {}))
    return jsonify({'ok': True})


# ── Batch mode routes ──

@app.route('/batch/start', methods=['POST'])
def batch_start():
    data = request.json
    judge_name = data['judge_name']

    # Assign judge a numeric ID based on order of appearance
    all_results = load_results()
    known_judges = sorted(set(r.get('judge_name', '') for r in all_results))
    if judge_name in known_judges:
        judge_id = known_judges.index(judge_name)
    else:
        judge_id = len(known_judges)

    n_gens = batch_data["metadata"]["n_generations"]
    pairings = get_pairings_for_judge(judge_id, batch_data["questions"], n_gens)

    # Figure out how many this judge has already completed
    judge_results = [r for r in all_results if r.get('judge_name') == judge_name]
    completed_qids = set(r['question_id'] for r in judge_results)
    resume_from = 0
    for i, p in enumerate(pairings):
        if p['question_id'] in completed_qids:
            resume_from = i + 1
        else:
            break

    return jsonify({
        "judge_id": judge_id,
        "pairings": pairings,
        "resume_from": resume_from,
    })


@app.route('/batch/pair', methods=['POST'])
def batch_pair():
    data = request.json
    q = batch_data["questions"][data["question_idx"]]
    resp_a = q["responses_a"][data["a_gen_idx"]]
    resp_b = q["responses_b"][data["b_gen_idx"]]

    if data["left_is_a"]:
        return jsonify({"left": resp_a, "right": resp_b})
    else:
        return jsonify({"left": resp_b, "right": resp_a})


# ── Interactive mode routes ──

interactive_assignment = {}

@app.route('/generate', methods=['POST'])
def generate_interactive():
    data = request.json
    prompt = data['prompt']

    if random.random() < 0.5:
        interactive_assignment['left'] = 'a'
        interactive_assignment['right'] = 'b'
    else:
        interactive_assignment['left'] = 'b'
        interactive_assignment['right'] = 'a'

    response_a = generate_from_session(sessions['a'], prompt)
    response_b = generate_from_session(sessions['b'], prompt)

    responses = {'a': response_a, 'b': response_b}
    return jsonify({
        'left': responses[interactive_assignment['left']],
        'right': responses[interactive_assignment['right']]
    })


# ── Shared routes ──

@app.route('/vote', methods=['POST'])
def vote():
    data = request.json
    choice = data['choice']

    if batch_mode:
        left_is_a = data['left_is_a']
        if choice == 'tie':
            winner_id = 'tie'
        elif (choice == 'left' and left_is_a) or (choice == 'right' and not left_is_a):
            winner_id = 'a'
        else:
            winner_id = 'b'

        result = {
            'timestamp': datetime.now().isoformat(),
            'judge_name': data['judge_name'],
            'question_id': data['question_id'],
            'question_idx': data['question_idx'],
            'a_gen_idx': data['a_gen_idx'],
            'b_gen_idx': data['b_gen_idx'],
            'left_is_a': left_is_a,
            'choice': choice,
            'winner': winner_id,
        }
        save_result(result)

        left_label = "Model A" if left_is_a else "Model B"
        right_label = "Model B" if left_is_a else "Model A"
        winner_label = "Model A" if winner_id == 'a' else ("Model B" if winner_id == 'b' else "Tie")

    else:
        if choice == 'tie':
            winner_id = 'tie'
        else:
            winner_id = interactive_assignment[choice]

        result = {
            'timestamp': datetime.now().isoformat(),
            'prompt': data.get('prompt', ''),
            'left_model': model_labels[interactive_assignment['left']],
            'right_model': model_labels[interactive_assignment['right']],
            'winner': winner_id,
        }
        save_result(result)

        left_label = model_labels[interactive_assignment['left']]
        right_label = model_labels[interactive_assignment['right']]
        winner_label = model_labels.get(winner_id, 'Tie')

    return jsonify({
        'winner': winner_id,
        'winner_id': winner_id,
        'winner_side': choice.capitalize() if choice != 'tie' else 'Tie',
        'winner_label': winner_label,
        'left_label': left_label,
        'right_label': right_label,
    })


@app.route('/stats')
def stats():
    all_results = load_results()
    a_wins = sum(1 for r in all_results if r['winner'] == 'a')
    b_wins = sum(1 for r in all_results if r['winner'] == 'b')
    ties = sum(1 for r in all_results if r['winner'] == 'tie')
    judge_count = len(set(r.get('judge_name', 'anon') for r in all_results))
    return jsonify({
        'total': len(all_results),
        'a_wins': a_wins,
        'b_wins': b_wins,
        'ties': ties,
        'judge_count': judge_count,
        'model_a': model_labels.get('a', batch_data["metadata"]["model_a"] if batch_data else '?'),
        'model_b': model_labels.get('b', batch_data["metadata"]["model_b"] if batch_data else '?'),
    })


# ─── Main ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLLM Blind A/B Comparison')
    parser.add_argument('--model-a', type=str, help='Path to model A directory')
    parser.add_argument('--model-b', type=str, help='Path to model B directory')
    parser.add_argument('--port', type=int, default=8400, help='Port (default 8400)')

    # Batch generation
    parser.add_argument('--generate-batch', action='store_true',
                        help='Generate batch eval responses (requires --model-a, --model-b, --questions)')
    parser.add_argument('--questions', type=str, default='configs/eval_questions_1.2B.json',
                        help='Path to questions JSON')
    parser.add_argument('--generations', type=int, default=3,
                        help='Number of generations per model per question (default 3)')
    parser.add_argument('--output', type=str, default='logs/batch_eval.json',
                        help='Output path for generated batch (default logs/batch_eval.json)')

    # Batch serving
    parser.add_argument('--batch', type=str,
                        help='Path to pre-generated batch JSON (enables batch eval mode)')

    args = parser.parse_args()

    if args.generate_batch:
        # ── Generate batch responses ──
        if not args.model_a or not args.model_b:
            parser.error("--generate-batch requires --model-a and --model-b")

        print(f"Generating batch eval responses...")
        print(f"  Model A: {args.model_a}")
        print(f"  Model B: {args.model_b}")
        print(f"  Questions: {args.questions}")
        print(f"  Generations per model: {args.generations}")
        print()

        batch = generate_batch(
            args.model_a, args.model_b, args.questions,
            n_generations=args.generations
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(batch, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"\nBatch saved to {output_path}")
        n_q = len(batch["questions"])
        n_g = batch["metadata"]["n_generations"]
        print(f"  {n_q} questions x {n_g} generations x 2 models = {n_q * n_g * 2} total responses")
        print(f"  {n_q * n_g * n_g} possible pairings per judge")

    elif args.batch:
        # ── Serve batch eval ──
        batch_mode = True
        batch_path = Path(args.batch)
        if not batch_path.exists():
            parser.error(f"Batch file not found: {args.batch}")
        batch_data = json.loads(batch_path.read_text(encoding='utf-8'))
        results_file = Path("logs/ab_batch_results.json")
        demographics_file = Path("logs/ab_demographics.json")

        meta = batch_data["metadata"]
        print(f"Batch Eval Mode")
        print(f"  Model A: {meta['model_a']}")
        print(f"  Model B: {meta['model_b']}")
        print(f"  Questions: {len(batch_data['questions'])}")
        print(f"  Generations per model: {meta['n_generations']}")
        print(f"  Generated: {meta['generated_at']}")
        print(f"\nStarting server on http://localhost:{args.port}")
        print(f"Results will be saved to {results_file}")
        app.run(host='0.0.0.0', port=args.port, debug=False)

    else:
        # ── Interactive mode ──
        if not args.model_a or not args.model_b:
            parser.error("Interactive mode requires --model-a and --model-b")

        batch_mode = False
        results_file = Path("logs/ab_results.json")
        demographics_file = Path("logs/ab_demographics.json")
        model_labels['a'] = Path(args.model_a).name
        model_labels['b'] = Path(args.model_b).name

        print(f"Interactive Mode")
        print(f"  Model A: {model_labels['a']}")
        print(f"  Model B: {model_labels['b']}")
        print(f"  (Labels hidden from users)")
        print()

        sessions['a'] = load_model(args.model_a)
        sessions['b'] = load_model(args.model_b)

        print(f"\nStarting server on http://localhost:{args.port}")
        print(f"Results will be saved to {results_file}")
        app.run(host='0.0.0.0', port=args.port, debug=False)
