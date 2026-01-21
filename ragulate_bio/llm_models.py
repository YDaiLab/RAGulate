"""
Utilities for loading and interacting with HuggingFace causal
language models.

This module wraps the somewhat complex model initialisation logic
from the notebook.  It hides the differences between BioGPT,
Mistral, Llama 3.1, Phi-3 and Qwen 2.5, and it ensures that
tokenisers are compatible with their models.  The helper
functions return cached tuples of ``(tokeniser, model)`` that can
be reused across calls.  Generation functions are also provided
for single and batched prompts.
"""

import os
from typing import Tuple, Optional, List, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from . import config
from .state import _SINGLETONS


def _ensure_pad_token_and_resize(tok: AutoTokenizer, mdl: AutoModelForCausalLM) -> None:
    """Ensure that the tokenizer has a pad token and resize embeddings if necessary."""
    added = False
    # Prefer reusing EOS as PAD; only add a new token if absolutely needed
    if tok.pad_token_id is None:
        if getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "[PAD]"})
            added = True
    # Left padding tends to be safer for chat models
    try:
        tok.padding_side = "left"
    except Exception:
        pass
    # Reflect pad in model config
    if getattr(mdl.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        mdl.config.pad_token_id = tok.pad_token_id
    # Only resize if we actually added a brand-new token above
    if added:
        mdl.resize_token_embeddings(len(tok))
        mdl.config.vocab_size = len(tok)


def _sync_special_tokens(tok: AutoTokenizer, cfg: Any) -> bool:
    """Add any special tokens declared in a model config that the tokenizer lacks."""
    specials: List[str] = []
    for key in ("bos_token", "eos_token", "unk_token", "pad_token",
                "sep_token", "cls_token", "mask_token"):
        val = getattr(cfg, key, None)
        if isinstance(val, str):
            specials.append(val)
    addl = getattr(cfg, "additional_special_tokens", None) or []
    specials.extend([s for s in addl if isinstance(s, str)])
    vocab = tok.get_vocab()
    missing = [s for s in specials if s not in vocab]
    if missing:
        tok.add_special_tokens({"additional_special_tokens": missing})
        return True
    return False


def _load_hf_model(name: str, token: Optional[str] = None, force_fresh: bool = False) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Load a HuggingFace causal language model and its tokenizer."""
    is_phi3 = 'phi3' in name.lower()
    is_qwen = 'qwen' in name.lower()
    attn_impl = 'eager' if is_phi3 else 'sdpa'
    model = AutoModelForCausalLM.from_pretrained(
        name,
        token=token,
        dtype=torch.bfloat16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map='auto',
        attn_implementation=attn_impl,
        force_download=force_fresh,
        local_files_only=False,
        trust_remote_code=is_qwen,
    )
    rev = getattr(model.config, "_commit_hash", None) or getattr(model.config, "repo_revision", None) or "main"
    tok = AutoTokenizer.from_pretrained(
        name,
        token=token,
        revision=rev,
        use_fast=True,
        force_download=force_fresh,
        local_files_only=False,
        trust_remote_code=is_qwen,
    )
    # Ensure pad token exists and sync vocab sizes
    _ensure_pad_token_and_resize(tok, model)
    if _sync_special_tokens(tok, model.config):
        model.resize_token_embeddings(len(tok))
        model.config.vocab_size = len(tok)
    vt = len(tok)
    vm = model.get_input_embeddings().weight.shape[0]
    # For Qwen (like Phi-3), pad the tokenizer up to the model’s vocab size instead of shrinking
    if is_qwen and vt < vm:
        need = vm - vt
        tok.add_special_tokens({'additional_special_tokens': [f'<|extra_{i}|>' for i in range(need)]})
        model.resize_token_embeddings(len(tok))
        vt = len(tok)
        vm = model.get_input_embeddings().weight.shape[0]
        model.config.vocab_size = vt
    elif is_phi3 and vt < vm:
        need = vm - vt
        tok.add_special_tokens({'additional_special_tokens': [f'<|extra_{i}|>' for i in range(need)]})
        model.resize_token_embeddings(len(tok))
        vt = len(tok)
        vm = model.get_input_embeddings().weight.shape[0]
        model.config.vocab_size = vt
    else:
        if vt < vm:
            model.resize_token_embeddings(vt)
            vm = model.get_input_embeddings().weight.shape[0]
            model.config.vocab_size = vt
        elif vt > vm:
            model.resize_token_embeddings(vt)
            vm = model.get_input_embeddings().weight.shape[0]
            model.config.vocab_size = vt
    assert vt == vm == model.config.vocab_size, f'Vocab mismatch for {name}: tok={vt}, mdl={vm}, cfg={model.config.vocab_size}'
    model.eval()
    return tok, model


def get_biogpt(name: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Return a cached BioGPT model and tokenizer."""
    model_name = name or config.BIOGPT_MODEL_NAME
    reg = _SINGLETONS['biogpt']
    if reg['obj'] is not None and reg['name'] == model_name:
        return reg['obj']
    if config.VERBOSE >= 1:
        print(f"[init] BioGPT -> {model_name}")
    tok, mdl = _load_hf_model(model_name)
    # Ensure the tokenizer's max length matches the model's context window
    if hasattr(mdl.config, "max_position_embeddings"):
        tok.model_max_length = mdl.config.max_position_embeddings
    reg['name'], reg['obj'] = model_name, (tok, mdl)
    if config.VERBOSE >= 1:
        print('[ready] BioGPT initialized')
    return reg['obj']


def get_mistral(name: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Return a cached Mistral model and tokenizer."""
    model_name = name or config.MISTRAL_MODEL_NAME
    reg = _SINGLETONS['mistral']
    if reg['obj'] is not None and reg['name'] == model_name:
        return reg['obj']
    if config.VERBOSE >= 1:
        print(f"[init] Mistral -> {model_name}")
    tok, mdl = _load_hf_model(model_name)
    reg['name'], reg['obj'] = model_name, (tok, mdl)
    if config.VERBOSE >= 1:
        print('[ready] Mistral initialized')
    return reg['obj']


def get_llama31(name: Optional[str] = None, token: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Return a cached Llama 3.1 model and tokenizer.

    An access token is required to download Meta’s Llama models.  If
    ``token`` is ``None`` the environment variable
    ``HUGGINGFACE_HUB_TOKEN`` is used.  If no token can be found
    this function will raise ``RuntimeError``.
    """
    model_name = name or config.LLAMA31_MODEL_NAME
    reg = _SINGLETONS['llama31']

    # Reuse cached model if already loaded with same name
    if reg['obj'] is not None and reg['name'] == model_name:
        return reg['obj']

    if config.VERBOSE >= 1:
        print(f"[init] Llama3.1 -> {model_name}")

    token = token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "No HF token set for meta-llama/Meta-Llama-3.1-8B-Instruct. "
            "Please set HUGGINGFACE_HUB_TOKEN or HF_TOKEN."
        )

    try:
        tok, mdl = _load_hf_model(model_name, token=token)
        reg['name'], reg['obj'] = model_name, (tok, mdl)
        if config.VERBOSE >= 1:
            print("[ready] Llama3.1 initialized")
        return reg['obj']
    except Exception as e:
        raise RuntimeError(f"Failed to initialise Llama3.1 ({model_name}): {e}")


def get_phi3(name: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Return a cached Phi-3 model and tokenizer."""
    model_name = name or config.PHI3_MODEL_NAME
    reg = _SINGLETONS['phi3']
    if reg['obj'] is not None and reg['name'] == model_name:
        return reg['obj']
    if config.VERBOSE >= 1:
        print(f"[init] Phi-3 -> {model_name}")
    tok, mdl = _load_hf_model(model_name)
    reg['name'], reg['obj'] = model_name, (tok, mdl)
    if config.VERBOSE >= 1:
        print('[ready] Phi-3 initialized')
    return reg['obj']


def get_qwen25(name: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Return a cached Qwen 2.5 model and tokenizer."""
    model_name = name or config.QWEN25_MODEL_NAME
    reg = _SINGLETONS['qwen25']
    if reg['obj'] is not None and reg['name'] == model_name:
        return reg['obj']
    if config.VERBOSE >= 1:
        print(f"[init] Qwen-2.5 -> {model_name}")
    tok, mdl = _load_hf_model(model_name)
    reg['name'], reg['obj'] = model_name, (tok, mdl)
    if config.VERBOSE >= 1:
        print('[ready] Qwen-2.5 initialized')
    return reg['obj']


def _model_max_ctx(tok: AutoTokenizer, mdl: AutoModelForCausalLM, default: int = 8192) -> int:
    """Return the maximum context window for a model.

    This examines both the tokenizer and the model config to find
    reasonable upper bounds on the number of tokens that can be
    encoded in one pass.  The ``default`` is used if no
    information is available.
    """
    a = getattr(tok, 'model_max_length', None)
    b = getattr(mdl.config, 'max_position_embeddings', None)
    cand = [x for x in (a, b) if isinstance(x, int) and 0 < x < 10000000]
    return max(cand) if cand else default


def _guard_pad_and_ctx(tok: AutoTokenizer, mdl: AutoModelForCausalLM) -> None:
    """Ensure the model has a pad token and left padding configured."""
    if getattr(tok, 'pad_token', None) is None:
        if getattr(tok, 'eos_token', None):
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({'pad_token': '[PAD]'})
        try:
            mdl.resize_token_embeddings(len(tok))
        except Exception:
            pass
    try:
        tok.padding_side = 'left'
    except Exception:
        pass
    try:
        if mdl.config.pad_token_id is None and tok.pad_token_id is not None:
            mdl.config.pad_token_id = tok.pad_token_id
    except Exception:
        pass


def _llm_generate(tok: AutoTokenizer, mdl: AutoModelForCausalLM, prompt: str, max_new_tokens: int = config.GEN_MAX_NEW) -> str:
    """Generate a single answer from a prompt."""
    _guard_pad_and_ctx(tok, mdl)
    max_ctx = _model_max_ctx(tok, mdl)
    max_input_tokens = max(8, max_ctx - max_new_tokens - config.SAFETY_MARGIN)
    inputs = tok(prompt, return_tensors='pt', padding=True, truncation=True, max_length=max_input_tokens).to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0], skip_special_tokens=True)



def _llm_generate_batch(
    tok,
    mdl,
    prompts,
    max_new_tokens: int = 8,
    *,
    mode: str = "yesno",  # "yesno" (old behavior) or "raw"
):
    """
    Generate continuations for a batch of prompts and return ONLY the
    newly generated text (not the prompt itself).

    When config.LLM_DEBUG is True, this function will print a small
    sample of (prompt, output) pairs to stdout for sanity checking.

    Additionally, outputs are normalised to a bare 'yes' / 'no'
    token where possible. Anything else becomes an empty string,
    which downstream is mapped to 0.5 (uncertain).

    mode = "yesno":
        - Return a list of 'yes' / 'no' / '' (for uncertain).
        - This preserves the original classifier behaviour.

    mode = "raw":
        - Return the raw generated text (without any yes/no normalisation).
    """
    if tok is None or mdl is None:
        raise ValueError("Tokenizer/model not initialised")

    if isinstance(prompts, str):
        prompts = [prompts]

    enc = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(mdl.device)

    input_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        gen_ids = mdl.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id if tok.pad_token_id is None else tok.pad_token_id,
        )

    # Slice off the prompt portion -> keep only new tokens
    gen_only = gen_ids[:, input_len:]

    # Decode only newly generated tokens
    texts = tok.batch_decode(gen_only, skip_special_tokens=True)
    raw_outputs: List[str] = [t.strip() for t in texts]

    # ----------------------------------------------------------
    # mode="raw": just return text as-is (for summaries, etc.)
    # ----------------------------------------------------------
    if mode == "raw":
        outputs = raw_outputs

    # ----------------------------------------------------------
    # mode="yesno": ORIGINAL BEHAVIOUR
    # ----------------------------------------------------------
    elif mode == "yesno":
        cleaned: List[str] = []
        for text in raw_outputs:
            s = (text or "").strip().lower()
            if not s:
                cleaned.append("")
                continue

            first = s.split()[0]

            if first in {"yes", "'yes'", "yes.", "yes,"}:
                cleaned.append("yes")
            elif first in {"no", "'no'", "no.", "no,"}:
                cleaned.append("no")
            else:
                cleaned.append("")  # unparseable -> will map to 0.5

        outputs = cleaned
    else:
        raise ValueError(f"Unknown mode for _llm_generate_batch: {mode!r}")

    # DEBUG block can stay the same, just use 'outputs'
    if getattr(config, "LLM_DEBUG", False):
        max_examples = getattr(config, "LLM_DEBUG_MAX_EXAMPLES", 5)
        max_prompt_chars = getattr(config, "LLM_DEBUG_MAX_PROMPT_CHARS", 400)

        print("\n[LLM DEBUG] ===== Batch outputs =====")
        for i, (p, out) in enumerate(zip(prompts, outputs)):
            if i >= max_examples:
                break
            p_str = "<None>" if p is None else str(p)
            if len(p_str) > max_prompt_chars:
                p_str = p_str[:max_prompt_chars] + "... [truncated]"

            print(f"\n--- Example {i} ---")
            print("[PROMPT]")
            print(p_str)
            print("[OUTPUT]")
            print(out if out else "<EMPTY STRING>")
            print("-" * 40)
        print("[LLM DEBUG] ==========================\n")

    return outputs


__all__ = [
    "get_biogpt",
    "get_mistral",
    "get_llama31",
    "get_phi3",
    "get_qwen25",
    "_llm_generate",
    "_llm_generate_batch",
    "_model_max_ctx",
    "_guard_pad_and_ctx",
]
