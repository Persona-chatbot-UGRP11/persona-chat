# persona-chat — Persona & Speech-Style Conditioned Korean Dialogue

> A research prototype that generates Korean chit-chat and then **re-styles it into a target persona's speech register**, so a conversational agent can "speak in character." Built as a DGIST UGRP (Undergraduate Group Research Program) team project, 2020.

## Motivation

Inspired by the science-fiction short story *관내분실* (Kim Choyeop), which imagines a "library" that stores the memories of the deceased so others can meet them again, this project asks a smaller, concrete version of the question: **can we make a conversational AI that talks in the style, tone, and personality of a specific (real or imaginary) character?**

We scoped it to an undergraduate-level prototype: generate a plausible Korean dialogue response, then impose a target character's **persona** — primarily their **speech register / tone of voice (말투)** — onto that response, so output stays *consistent in character* rather than generically neutral.

## Approach

The system separates **what to say** from **how to say it**:

```
user utterance ─▶ [ response generation ]  ─▶ neutral response
                                              │
                                              ▼
                        [ persona / style transfer ]  ─▶ in-character response
```

1. **Response generation** — a Korean chit-chat response is produced from a generative dialogue model (KoGPT2, fine-tuned on an ~11k-pair Korean Q&A chit-chat dataset; an earlier Seq2Seq baseline is also included).
2. **Persona / speech-style transfer** — a separate Seq2Seq (GRU + attention) network rewrites that response into a target speech register, concatenated onto the generator's output (`GPT2 → intermediate response → Seq2Seq → final styled response`). Target registers include **다나까체** (formal/declarative), **음슴체** (terse `-ㅁ` style), and **하오체** (archaic-polite).

We also explored persona-by-**genre**: training the generator on character dialogue from historical-drama (사극) vs. teen-drama (하이틴) scripts, including a **zero-shot variant** that prepends a genre label token (`[hist]` / `[high]`) so a single model conditions its style on the label.

## Models

- **Seq2Seq** — RNN encoder–decoder (GRU), Bahdanau-style attention, `Okt` (KoNLPy) tokenizer, teacher forcing. Used both as a dialogue baseline and as the style-transfer module (`seq2seq_wh.py`).
- **KoGPT2** — pretrained Korean GPT-2 (MXNet/GluonNLP), fine-tuned for chit-chat and used as the generation stage (`train.py`).

## Experiments & evaluation

| # | Experiment | Idea | Metrics |
|---|------------|------|---------|
| 1 | Seq2Seq on genre datasets | Learn a character's tone from 사극 / 하이틴 dialogue (separate models + zero-shot label token) | BLEU, perplexity |
| 2 | Style transfer — full sentence | GPT2→Seq2Seq concat; rewrite whole sentence into 다나까/음슴/하오 register | BLEU, loss |
| 3 | Style transfer — suffix only | Rewrite only the trailing eojeol(s), where Korean register lives | BLEU, loss |
| 4 | Style transfer — content words excluded | Use morpheme analysis to feed only non-noun tokens, isolating the register signal | BLEU, loss |

Custom parallel datasets were built for the style-transfer experiments (~11.8k pairs per register × 3 registers).

## Results & honest learnings

- **Genre persona (Exp 1) underperformed.** Even trained to overfitting, generation quality was poor. Root causes we identified: dialogue scraped from drama scripts was *raw/uncurated*, the corpus was *small* for a generative model, and a plain RNN encoder (no segmentation) degrades sharply on longer sentences — and drama lines skew long.
- **Speech-style transfer (Exp 2–4) worked noticeably better.** BLEU improved across epochs and the suffix/verb-focused variants concentrated the model on where register actually changes, which is the cleaner signal.
- **Takeaway:** factoring *content generation* from *style conditioning* was the right decomposition; the bottleneck was data quality/quantity and the RNN-era backbone, not the framing.

### What we'd do differently now

This was 2020, pre-instruction-tuned-LLM. Today the same goal is better served by an instruction-tuned LLM with persona/style **prompting** (or lightweight **LoRA** adapters per persona), curated parallel/preference data, and a proper evaluation harness (human + automatic) instead of BLEU alone. The separation of "content" and "persona/style" layers, however, still holds up as a design.

## Repository layout

```
.
├── train.py                 # KoGPT2 chit-chat generation + Seq2Seq style-transfer concat
├── seq2seq_wh.py            # Seq2Seq (GRU + attention, Okt) — dialogue & style transfer
├── train-Copy1.py           # training variant
├── KoGPT2_chatbot.ipynb     # KoGPT2 chatbot notebook
├── Chatbot_data/            # Korean Q&A chit-chat dataset (~11k pairs)
├── Seq2seq_data/            # parallel style-transfer data (다나까 / 음슴 / 하오)
├── chat_test_question/      # evaluation question sets
├── gpt_s2s_result.txt       # sample GPT2→Seq2Seq outputs
├── *.pt / *.pkl             # trained Seq2Seq / model weights
└── docs/
    └── UGRP_final_report.pdf  # full research report (Korean)
```

## Tech stack

Python · PyTorch (Seq2Seq) · MXNet / GluonNLP + KoGPT2 · KoNLPy (Okt) · SentencePiece · BLEU / perplexity.

## Report

The full research report (motivation, model survey of Seq2Seq/BERT/GPT-2, methods, experiments, and analysis) is in **[`docs/UGRP_final_report.pdf`](docs/UGRP_final_report.pdf)**.

---

*DGIST 2020 UGRP team project.*
