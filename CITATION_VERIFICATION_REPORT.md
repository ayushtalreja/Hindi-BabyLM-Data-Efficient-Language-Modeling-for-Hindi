# Citation Verification Report: methodology.tex

**Generated:** 2026-01-19
**Verified Against:** references.bib

---

## Summary

| Status | Count | Description |
|--------|-------|-------------|
| ✅ | 7 | Correct citations |
| ⚠️ | 4 | Partially correct / Minor inaccuracies |
| ❌ | 2 | Missing or wrong citations |

---

## Detailed Verification

### 1. `warstadt-etal-2023-findings` — BabyLM Challenge

**Location:** Line 4
**Methodology Claim:**
> "The implementation follows the BabyLM challenge framework, adapted specifically for Hindi linguistic phenomena. All experiments utilize approximately 10 and 100 million words, following the 'strict-small' and 'strict' track of the BabyLM challenge."

**Source Information (references.bib):**
```bibtex
@inproceedings{warstadt-etal-2023-findings,
    title     = {Findings of the {B}aby{LM} Challenge: Sample-Efficient Pretraining...},
    booktitle = {Proceedings of the BabyLM Challenge at the 27th Conference on CoNLL},
    year      = {2023}
}
```

**Verification:** ✅ **CORRECT**

The BabyLM Challenge 2023 indeed has:
- **Strict-small track:** 10M words
- **Strict track:** 100M words

**Sources:** [BabyLM 2023 Archive](https://babylm.github.io/archive_2023.html), [ACL Anthology](https://aclanthology.org/volumes/2023.conll-babylm/)

---

### 2. `kakwani-etal-2020-indicnlpsuite` — IndicCorp V2 & IndicGLUE

**Location:** Lines 153, 546
**Methodology Claims:**
> "IndicCorp V2 provides the largest proportion of the corpus, contributing formal written Hindi from news articles and web content."
> "IndicGLUE provides a benchmark suite of eight Hindi NLP tasks."

**Source Information (references.bib):**
```bibtex
@inproceedings{kakwani-etal-2020-indicnlpsuite,
    title     = {{I}ndic{NLPS}uite: Monolingual Corpora, Evaluation Benchmarks...},
    booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2020},
    year      = {2020}
}
```

**Verification:** ⚠️ **PARTIALLY CORRECT**

- ✅ IndicCorp claim is accurate
- ⚠️ The original IndicGLUE has these **core tasks**: Article Genre Classification, Headline Prediction, Wikipedia Section-Title Prediction, Cloze-style QA, Winograd NLI, COPA
- The methodology lists 8 tasks including "Movie Review Sentiment" and "Product Review Sentiment" which may be from separate datasets

**Suggested Revision:**
If sentiment tasks are from external sources (not IndicGLUE), add appropriate citations or clarify:
> "IndicGLUE provides the core benchmark tasks, supplemented with additional sentiment datasets."

**Sources:** [IndicNLPSuite Paper](https://aclanthology.org/2020.findings-emnlp.445/), [AI4Bharat](https://indicnlp.ai4bharat.org)

---

### 3. `indicdialogue` — IndicDialogue Dataset

**Location:** Line 157
**Methodology Claim:**
> "IndicDialogue provides conversational Hindi from movie subtitles, capturing spoken language patterns not found in written corpora."

**Source Information (references.bib):**
```bibtex
@article{indicdialogue,
    title   = {{IndicDialogue}: A Multi-domain Multi-lingual Dialogue Dataset...},
    journal = {Mendeley Data},
    year    = {2023}
}
```

**Verification:** ⚠️ **ACCEPTABLE BUT WEAK**

- The reference points to Mendeley Data, which is a data repository, not a peer-reviewed publication
- The description is accurate

**Suggested Revision:**
Consider finding a published paper describing this dataset or adding:
> "Available at Mendeley Data."

---

### 4. `pratham-books-2008-storyweaver` — StoryWeaver Children's Books

**Location:** Line 159
**Methodology Claim:**
> "Children's literature from StoryWeaver (Pratham Books) contributes developmentally appropriate text with simplified vocabulary and grammatical structures."

**Source Information (references.bib):**
```
NOT FOUND IN REFERENCES.BIB
```

**Verification:** ❌ **MISSING CITATION**

The citation `\cite{pratham-books-2008-storyweaver}` does not exist in references.bib. This will cause a LaTeX compilation error (undefined citation).

**Required Action:**
Add the following entry to references.bib:
```bibtex
@misc{pratham-books-storyweaver,
    title        = {{StoryWeaver}: Open-source Platform for Multilingual Children's Literature},
    author       = {{Pratham Books}},
    year         = {2015},
    howpublished = {\url{https://storyweaver.org.in/}},
    note         = {Accessed: 2026-01-19}
}
```

And update the citation key in methodology.tex from `pratham-books-2008-storyweaver` to `pratham-books-storyweaver`.

---

### 5. `broder1997resemblance` — MinHash LSH

**Location:** Line 185
**Methodology Claim:**
> "Pairs with estimated Jaccard similarity above a chosen threshold (e.g., 0.8) are treated as near-duplicates, following the MinHash-based resemblance framework of Broder."

**Source Information (references.bib):**
```bibtex
@inproceedings{broder1997resemblance,
    title     = {On the Resemblance and Containment of Documents},
    author    = {Broder, Andrei Z.},
    booktitle = {SEQUENCES '97},
    year      = {1997}
}
```

**Verification:** ✅ **CORRECT**

This is the seminal paper introducing MinHash for document similarity estimation.

---

### 6. `kudo-richardson-2018-sentencepiece` — SentencePiece Tokenizer

**Location:** Line 263
**Methodology Claim:**
> "SentencePiece implements a language-agnostic subword tokenization framework supporting both unigram language model and byte-pair encoding algorithms. It operates directly on raw text without language-specific pre-tokenization."

**Source Information (references.bib):**
```bibtex
@inproceedings{kudo-richardson-2018-sentencepiece,
    title     = {{S}entence{P}iece: A simple and language independent subword tokenizer...},
    booktitle = {Proceedings of EMNLP 2018: System Demonstrations},
    year      = {2018}
}
```

**Verification:** ✅ **CORRECT**

The claim accurately describes SentencePiece's key features:
- Language-agnostic
- Supports unigram LM and BPE
- Works directly on raw text without pre-tokenization

**Sources:** [SentencePiece Paper](https://aclanthology.org/D18-2012/)

---

### 7. `radford2019language` — GPT-2 Architecture

**Location:** Lines 265, 337
**Methodology Claims:**
> "For BPE, SentencePiece is configured to follow the GPT-style tokenization paradigm."
> "The GPT-2 implementation follows the decoder-only autoregressive architecture. The model predicts each token given all previous tokens."

**Source Information (references.bib):**
```bibtex
@article{radford2019language,
    title   = {Language Models are Unsupervised Multitask Learners},
    author  = {Radford, Alec and Wu, Jeffrey and Child, Rewon...},
    journal = {OpenAI Blog},
    year    = {2019}
}
```

**Verification:** ✅ **CORRECT**

GPT-2 is indeed:
- A decoder-only, autoregressive transformer
- Uses causal (unidirectional) attention
- Predicts next token given all previous tokens

**Sources:** [OpenAI GPT-2](https://github.com/openai/gpt-2), [GPT-2 Paper PDF](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

### 8. `devlin-etal-2019-bert` — WordPiece Tokenization

**Location:** Line 267
**Methodology Claim:**
> "WordPiece tokenization follows the BERT approach with **greedy frequency-based merging**."

**Source Information (references.bib):**
```bibtex
@inproceedings{devlin-etal-2019-bert,
    title     = {{BERT}: Pre-training of Deep Bidirectional Transformers...},
    booktitle = {Proceedings of NAACL-HLT 2019},
    year      = {2019}
}
```

**Verification:** ⚠️ **INACCURATE**

The claim contains an error:
- ❌ WordPiece does **NOT** use "frequency-based merging"
- WordPiece merges pairs that **maximize the likelihood of the training data**, not the most frequent pairs
- BPE uses frequency-based merging; WordPiece uses likelihood-based merging

**Suggested Revision:**
> "WordPiece tokenization follows the BERT approach with **greedy likelihood-based merging**, selecting pairs that maximize the training data likelihood rather than the most frequent pairs."

Or simply:
> "WordPiece tokenization follows the BERT approach with greedy subword merging."

**Sources:** [Hugging Face LLM Course - WordPiece](https://huggingface.co/learn/llm-course/en/chapter6/6), [WordPiece Explained](https://towardsdatascience.com/wordpiece-subword-based-tokenization-algorithm-1fbd14394ed7/)

---

### 9. `he2021deberta` — DeBERTa Architecture

**Location:** Line 357
**Methodology Claim:**
> "DeBERTa (Decoding-enhanced BERT with Disentangled Attention) implements masked language modeling with several architectural innovations. The key contribution is disentangled attention, which separately encodes content and position information through **three attention components**: content-to-content (how words attend to other words), content-to-position (how words attend to positions), and position-to-content (how positions influence word attention)."

**Source Information (references.bib):**
```bibtex
@inproceedings{he2021deberta,
    title     = {{DeBERTa}: Decoding-Enhanced {BERT} with Disentangled Attention},
    booktitle = {International Conference on Learning Representations},
    year      = {2021}
}
```

**Verification:** ⚠️ **PARTIALLY CORRECT**

- ✅ DeBERTa description is accurate
- ⚠️ The original paper describes **four** attention components: content-to-content (C2C), content-to-position (C2P), position-to-content (P2C), and **position-to-position (P2P)**
- However, P2P is often omitted in practice and many implementations use only three

**Suggested Revision (Optional):**
> "...through attention components including content-to-content, content-to-position, and position-to-content interactions."

(Removes the specific "three" to avoid potential discrepancy)

**Sources:** [DeBERTa Paper](https://arxiv.org/abs/2006.03654), [OpenReview](https://openreview.net/forum?id=XPZIaotutsD)

---

### 10. `loshchilov2019decoupled` — AdamW Optimizer

**Location:** Line 438
**Methodology Claim:**
> "Training employs the AdamW optimizer with decoupled weight decay."

**Source Information (references.bib):**
```bibtex
@inproceedings{loshchilov2019decoupled,
    title     = {Decoupled Weight Decay Regularization},
    booktitle = {7th International Conference on Learning Representations, ICLR 2019},
    year      = {2019}
}
```

**Verification:** ✅ **CORRECT**

The Loshchilov & Hutter paper introduces AdamW, which decouples weight decay from the gradient-based update, unlike L2 regularization in standard Adam.

**Sources:** [AdamW Paper](https://arxiv.org/abs/1711.05101), [OpenReview](https://openreview.net/forum?id=Bkg6RiCqY7)

---

### 11. `loshchilov2017sgdr` — Learning Rate Schedule

**Location:** Line 465
**Methodology Claim:**
> "A **cosine learning rate schedule with linear warmup** provides smooth convergence."

**Source Information (references.bib):**
```bibtex
@inproceedings{loshchilov2017sgdr,
    title     = {{SGDR:} Stochastic Gradient Descent with Warm Restarts},
    booktitle = {ICLR 2017},
    year      = {2017}
}
```

**Verification:** ⚠️ **INACCURATE**

The SGDR paper introduces:
- ✅ Cosine annealing schedule
- ❌ **NOT** linear warmup

**What SGDR actually proposes:** "Warm restarts" where the learning rate is reset to a high value periodically (not a gradual warmup from zero).

**Linear warmup** was popularized by:
- Vaswani et al. (2017) — "Attention Is All You Need" (Transformer)
- Devlin et al. (2019) — BERT

**Suggested Revision:**
> "A cosine learning rate schedule with linear warmup provides smooth convergence. The cosine decay follows \cite{loshchilov2017sgdr}, while the linear warmup is standard practice in transformer training \cite{vaswani2017attention}."

Or simply remove the citation from the warmup claim:
> "A cosine learning rate schedule (following \cite{loshchilov2017sgdr}) with linear warmup provides smooth convergence."

**Sources:** [SGDR Paper](https://arxiv.org/abs/1608.03983), [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

### 12. `wandb` — Weights & Biases

**Location:** Line 485
**Methodology Claim:**
> "Weights & Biases integration provides real-time monitoring of training dynamics."

**Source Information (references.bib):**
```bibtex
@misc{wandb,
    title  = {Experiment Tracking with Weights and Biases},
    author = {Biewald, Lukas},
    year   = {2020}
}
```

**Verification:** ✅ **CORRECT**

Standard reference for the W&B platform.

---

### 13. `jumelet-etal-2021-language` — MultiBLiMP

**Location:** Line 574
**Methodology Claim:**
> "MultiBLiMP evaluates grammatical competence through minimal pairs from the HuggingFace dataset `jumelet/multiblimp`."

**Source Information (references.bib):**
```bibtex
@inproceedings{jumelet-etal-2021-language,
    title     = {Language Models Use Monotonicity to Assess {NPI} Licensing},
    booktitle = {Findings of ACL-IJCNLP 2021},
    year      = {2021}
}
```

**Verification:** ❌ **WRONG CITATION**

**Critical Error:** The cited paper is about **NPI (Negative Polarity Item) Licensing**, NOT MultiBLiMP!

MultiBLiMP is a **separate work** by Jumelet et al., published in **2025**:
- Title: "MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs"
- Covers 101 languages with 125,000+ minimal pairs

**Required Action:**
Add the correct citation to references.bib:
```bibtex
@article{jumelet2025multiblimp,
    title   = {MultiBLiMP 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs},
    author  = {Jumelet, Jaap and Weissweiler, Leonie and Bisazza, Arianna},
    journal = {arXiv preprint arXiv:2504.02768},
    year    = {2025},
    url     = {https://arxiv.org/abs/2504.02768}
}
```

And update the citation in methodology.tex:
```latex
% FROM:
MultiBLiMP evaluates grammatical competence... \cite{jumelet-etal-2021-language}

% TO:
MultiBLiMP evaluates grammatical competence... \cite{jumelet2025multiblimp}
```

**Sources:** [MultiBLiMP Paper](https://arxiv.org/abs/2504.02768), [HuggingFace Dataset](https://huggingface.co/datasets/jumelet/multiblimp), [GitHub](https://github.com/jumelet/multiblimp)

---

## Action Items Summary

### Critical (Must Fix)

1. **Add missing `pratham-books-storyweaver` citation** to references.bib
2. **Replace `jumelet-etal-2021-language` with `jumelet2025multiblimp`** for MultiBLiMP

### Recommended (Should Fix)

3. **Correct WordPiece description** (Line 267): Change "greedy frequency-based merging" to "greedy likelihood-based merging" or remove the specific merging type
4. **Clarify cosine LR + warmup citation** (Line 465): SGDR doesn't include linear warmup; either add a separate citation for warmup or modify the phrasing

### Optional (Minor Improvements)

5. Consider clarifying which IndicGLUE tasks are from the original paper vs. supplementary datasets
6. Consider updating DeBERTa attention description to acknowledge four components (though current description is acceptable)

---

## New References to Add

```bibtex
% StoryWeaver - MISSING REFERENCE
@misc{pratham-books-storyweaver,
    title        = {{StoryWeaver}: Open-source Platform for Multilingual Children's Literature},
    author       = {{Pratham Books}},
    year         = {2015},
    howpublished = {\url{https://storyweaver.org.in/}},
    note         = {Accessed: 2026-01-19}
}

% MultiBLiMP - CORRECT REFERENCE (replaces NPI paper)
@article{jumelet2025multiblimp,
    title   = {{MultiBLiMP} 1.0: A Massively Multilingual Benchmark of Linguistic Minimal Pairs},
    author  = {Jumelet, Jaap and Weissweiler, Leonie and Bisazza, Arianna},
    journal = {arXiv preprint arXiv:2504.02768},
    year    = {2025},
    url     = {https://arxiv.org/abs/2504.02768}
}

% Optional: Linear warmup citation
@inproceedings{vaswani2017attention,
    author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
    title     = {Attention is All You Need},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2017}
}
```

---

## Verification Sources

- [BabyLM Challenge 2023](https://babylm.github.io/archive_2023.html)
- [IndicNLPSuite (ACL Anthology)](https://aclanthology.org/2020.findings-emnlp.445/)
- [SentencePiece (ACL Anthology)](https://aclanthology.org/D18-2012/)
- [GPT-2 (OpenAI)](https://github.com/openai/gpt-2)
- [BERT (ACL Anthology)](https://aclanthology.org/N19-1423/)
- [DeBERTa (arXiv)](https://arxiv.org/abs/2006.03654)
- [AdamW (arXiv)](https://arxiv.org/abs/1711.05101)
- [SGDR (arXiv)](https://arxiv.org/abs/1608.03983)
- [MultiBLiMP (arXiv)](https://arxiv.org/abs/2504.02768)
- [Hugging Face WordPiece Tutorial](https://huggingface.co/learn/llm-course/en/chapter6/6)
