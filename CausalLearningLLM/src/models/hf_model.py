"""
HuggingFace model wrapper for the CRA framework.

Supports both causal LM (GPT-2, BioGPT, gpt2-medium) and encoder-only
(BERT, Bio_ClinicalBERT) architectures. Encoder-only models skip causal
answer scoring; representation extraction and intervention hooks work for both.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoModel, AutoModelForMaskedLM, AutoTokenizer

logger = logging.getLogger(__name__)

# BERT-family model name substrings that need encoder-only loading
_ENCODER_ONLY_PATTERNS = ["bert", "roberta", "electra", "deberta", "albert"]


def _is_encoder_only(model_name: str) -> bool:
    name_lower = model_name.lower()
    return any(p in name_lower for p in _ENCODER_ONLY_PATTERNS)


class ClinicalLLMWrapper:
    """Thin wrapper around a HuggingFace model supporting causal-LM and encoder-only."""

    def __init__(
        self,
        model_name: str = "distilgpt2",
        tokenizer_name: Optional[str] = None,
        device: str = "auto",
        dtype: str = "float32",
        max_length: int = 512,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.max_length = max_length
        self.load_in_8bit = load_in_8bit
        self.trust_remote_code = trust_remote_code
        self.is_encoder_only = _is_encoder_only(model_name)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        _dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = _dtype_map.get(dtype, torch.float32)

        self.model: Optional[nn.Module] = None
        self.tokenizer = None
        self._hooks: List[Any] = []
        self.n_layers: int = 0
        self.hidden_size: int = 0
        self._mlm_model: Optional[nn.Module] = None
        self._mlm_available: Optional[bool] = None  # None = untried

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_tokenizer(self) -> "ClinicalLLMWrapper":
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self

    def load_model(self) -> "ClinicalLLMWrapper":
        logger.info("Loading model: %s (encoder_only=%s)", self.model_name, self.is_encoder_only)
        kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
        }
        if self.dtype in (torch.float16, torch.bfloat16):
            kwargs["torch_dtype"] = self.dtype

        if self.is_encoder_only:
            self.model = AutoModel.from_pretrained(self.model_name, **kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)

        self.model.eval()
        self.model.to(self.device)

        cfg = self.model.config
        self.n_layers = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 6))
        self.hidden_size = getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 768))
        logger.info("Model loaded | layers=%d hidden=%d device=%s", self.n_layers, self.hidden_size, self.device)
        return self

    def load(self) -> "ClinicalLLMWrapper":
        self.load_tokenizer()
        self.load_model()
        return self

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        texts: List[str],
        output_hidden_states: bool = True,
        output_attentions: bool = False,
    ) -> Dict[str, Any]:
        inputs = self.tokenize_batch(texts)
        outputs = self.model(
            **inputs,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        logits = getattr(outputs, "logits", None)
        return {
            "logits": logits,
            "hidden_states": outputs.hidden_states if output_hidden_states else None,
            "attentions": outputs.attentions if output_attentions else None,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs.get("attention_mask"),
        }

    # ------------------------------------------------------------------
    # Hidden-state extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_hidden_states(
        self,
        texts: List[str],
        layers: Optional[List[int]] = None,
        pooling: str = "mean",
    ) -> Dict[int, np.ndarray]:
        """Return {layer_idx: np.ndarray [batch, hidden]}."""
        outputs = self.forward(texts, output_hidden_states=True)
        all_hidden = outputs["hidden_states"]       # tuple len = n_layers+1
        attention_mask = outputs["attention_mask"]

        if layers is None:
            layers = list(range(len(all_hidden)))

        result: Dict[int, np.ndarray] = {}
        for l in layers:
            if l >= len(all_hidden):
                continue
            h = all_hidden[l]  # [batch, seq, hidden]
            pooled = self._pool(h, attention_mask, pooling)
            result[l] = pooled.cpu().float().numpy()
        return result

    def _pool(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        method: str,
    ) -> torch.Tensor:
        if method == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return hidden.mean(1)
        elif method == "last":
            if attention_mask is not None:
                lengths = attention_mask.sum(-1).long() - 1
                batch = hidden.size(0)
                return hidden[torch.arange(batch, device=hidden.device), lengths]
            return hidden[:, -1]
        elif method == "max":
            return hidden.max(1).values
        return hidden.mean(1)

    # ------------------------------------------------------------------
    # Answer scoring
    # ------------------------------------------------------------------

    def _get_mlm_model(self) -> bool:
        """Lazily load AutoModelForMaskedLM for PLL scoring. Returns True if available."""
        if self._mlm_available is not None:
            return self._mlm_available
        try:
            logger.info("Loading MLM head for PLL scoring: %s", self.model_name)
            kwargs: Dict[str, Any] = {"trust_remote_code": self.trust_remote_code}
            if self.dtype in (torch.float16, torch.bfloat16):
                kwargs["torch_dtype"] = self.dtype
            self._mlm_model = AutoModelForMaskedLM.from_pretrained(self.model_name, **kwargs)
            self._mlm_model.eval()
            self._mlm_model.to(self.device)
            self._mlm_available = True
        except Exception as e:
            logger.warning("AutoModelForMaskedLM not available for %s: %s", self.model_name, e)
            self._mlm_available = False
        return self._mlm_available

    @torch.no_grad()
    def _score_pll(self, context_ids: torch.Tensor, answer_start_idx: int) -> float:
        """Pseudo-log-likelihood score (Wang & Cho 2019) for answer tokens.

        Args:
            context_ids: 1-D tensor of token ids for the full (question + answer) sequence.
            answer_start_idx: index of the first answer token in context_ids.

        Returns:
            Sum of log P(token_i | context with token_i masked) over answer tokens.
        """
        mlm = self._mlm_model
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            return 0.0

        seq_len = context_ids.size(0)
        ans_len = seq_len - answer_start_idx
        n_tokens = min(ans_len, 10)  # limit for speed

        total = 0.0
        for i in range(n_tokens):
            pos = answer_start_idx + i
            masked = context_ids.clone()
            masked[pos] = mask_token_id
            input_ids = masked.unsqueeze(0).to(self.device)  # [1, seq]
            out = mlm(input_ids=input_ids)
            log_probs = torch.nn.functional.log_softmax(out.logits[0, pos], dim=-1)
            total += log_probs[context_ids[pos]].item()
        return total

    @torch.no_grad()
    def _score_cls_similarity(self, text: str, answer_choice: str, all_choices: List[str]) -> float:
        """Fallback: cosine similarity of [CLS] embedding of (text+choice) vs mean of all choices."""
        def cls_embed(t: str) -> torch.Tensor:
            enc = self.tokenize_batch([t])
            out = self.model(**enc, output_hidden_states=True)
            # hidden_states[0] is the embedding layer output; use last hidden state via index -1
            hs = out.hidden_states[-1]  # [1, seq, hidden]
            return hs[0, 0]  # [CLS] token

        target_emb = cls_embed(text + " " + answer_choice)
        all_embs = torch.stack([cls_embed(text + " " + c) for c in all_choices])  # [C, hidden]
        mean_emb = all_embs.mean(0)
        cos = torch.nn.functional.cosine_similarity(target_emb.unsqueeze(0), mean_emb.unsqueeze(0))
        return cos.item()

    @torch.no_grad()
    def score_answer_choice(self, text: str, answer_choice: str, all_choices: Optional[List[str]] = None) -> float:
        """Score an answer choice given a context text.

        For causal LM: sum of log P(token | prefix) over answer tokens.
        For encoder-only: PLL scoring via masked LM, or CLS-similarity fallback.
        """
        if not self.is_encoder_only:
            # Causal LM path (unchanged)
            full = text + " " + answer_choice
            inputs = self.tokenize_batch([full])
            out = self.model(**inputs)
            logits = out.logits[0]  # [seq, vocab]

            text_ids = self.tokenizer.encode(text, add_special_tokens=False)
            ans_ids = self.tokenizer.encode(" " + answer_choice, add_special_tokens=False)
            if not ans_ids:
                return 0.0

            lp = torch.nn.functional.log_softmax(logits, dim=-1)
            total = 0.0
            start = len(text_ids)
            for i, tok in enumerate(ans_ids):
                pos = start + i - 1
                if pos < lp.size(0):
                    total += lp[pos, tok].item()
            return total

        # Encoder-only: PLL path
        if self._get_mlm_model():
            full = text + " " + answer_choice
            enc = self.tokenizer(
                full,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            context_ids = enc["input_ids"][0]  # 1-D
            # Determine where answer tokens start in the tokenised sequence
            text_only_enc = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                add_special_tokens=True,
            )
            answer_start_idx = text_only_enc["input_ids"].size(1) - 1  # last tok of text overlaps
            # Clamp to valid range
            answer_start_idx = max(1, min(answer_start_idx, context_ids.size(0) - 1))
            return self._score_pll(context_ids, answer_start_idx)
        else:
            # CLS-similarity fallback
            choices = all_choices if all_choices else [answer_choice]
            return self._score_cls_similarity(text, answer_choice, choices)

    @torch.no_grad()
    def get_answer_probabilities(
        self,
        texts: List[str],
        answer_choices: List[str],
    ) -> Dict[str, np.ndarray]:
        """Normalised probability of each choice for each text."""
        raw: Dict[str, List[float]] = {c: [] for c in answer_choices}
        for text in texts:
            for choice in answer_choices:
                raw[choice].append(self.score_answer_choice(text, choice, all_choices=answer_choices))

        mat = np.stack([np.array(raw[c]) for c in answer_choices], axis=1)  # [N, C]
        mat -= mat.max(1, keepdims=True)
        exp = np.exp(mat)
        probs = exp / exp.sum(1, keepdims=True)
        return {c: probs[:, i] for i, c in enumerate(answer_choices)}

    @torch.no_grad()
    def predict(
        self,
        texts: List[str],
        answer_choices: List[str],
    ) -> Tuple[List[str], np.ndarray]:
        probs = self.get_answer_probabilities(texts, answer_choices)
        mat = np.stack([probs[c] for c in answer_choices], axis=1)
        idx = mat.argmax(1)
        return [answer_choices[i] for i in idx], mat

    # ------------------------------------------------------------------
    # Model-space intervention via hooks
    # ------------------------------------------------------------------

    def register_intervention_hook(
        self,
        layer: int,
        intervention_fn: Callable,
    ) -> None:
        """Register a hook that modifies [batch, seq, hidden] activations."""
        self._remove_hooks()
        layers = self._get_transformer_layers()
        if not layers or layer >= len(layers):
            logger.warning("Layer %d out of range (max %d)", layer, len(layers) - 1)
            return

        def _hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            h_np = h.detach().cpu().float().numpy()
            h_int = intervention_fn(h_np)
            h_new = torch.tensor(h_int, dtype=h.dtype, device=h.device)
            if isinstance(output, tuple):
                return (h_new,) + output[1:]
            return h_new

        handle = layers[layer].register_forward_hook(_hook)
        self._hooks.append(handle)

    def _remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _get_transformer_layers(self) -> List[nn.Module]:
        m = self.model
        # GPT-2 style: model.transformer.h
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return list(m.transformer.h)
        # BERT / RoBERTa: model.bert.encoder.layer or model.encoder.layer
        if hasattr(m, "bert") and hasattr(m.bert, "encoder") and hasattr(m.bert.encoder, "layer"):
            return list(m.bert.encoder.layer)
        if hasattr(m, "roberta") and hasattr(m.roberta, "encoder") and hasattr(m.roberta.encoder, "layer"):
            return list(m.roberta.encoder.layer)
        if hasattr(m, "encoder") and hasattr(m.encoder, "layer"):
            return list(m.encoder.layer)
        # BioGPT: model.biogpt.layers
        if hasattr(m, "biogpt") and hasattr(m.biogpt, "layers"):
            return list(m.biogpt.layers)
        # LLaMA / Mistral / OPT-style: model.model.layers
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return list(m.model.layers)
        # Fallback: find first ModuleList with multiple children
        for _, mod in m.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 1:
                return list(mod)
        return []

    def __del__(self) -> None:
        self._remove_hooks()
        self._mlm_model = None
