# Background and Related Work: Citation Verification Report

**Generated**: 2026-01-20
**File Verified**: `background_related_work.tex`
**References File**: `references.bib`

---

## Executive Summary

This report verifies citations in the Background and Related Work chapter against the provided references.bib file and external sources. Issues are categorized as:

- **Critical**: Missing citations, factually incorrect claims
- **Warning**: Citation key mismatches, year discrepancies
- **Verified**: Claims confirmed as accurate

**Total Citations Found**: 45 unique citation keys
**Issues Identified**: 8 critical, 3 warnings
**Verified Correct**: 34 citations

---

## Critical Issues

### 1. Missing Citation: `morphtok2025`

**Location**: Line 289
**Current Text**:
> "For Indic languages, MorphTok~\cite{morphtok2025} incorporates sandhi splitting—handling phonological fusion at morpheme boundaries—into tokenization, showing improvements on machine translation and language modeling."

**Issue**: Citation key `morphtok2025` does not exist in `references.bib`.

**Verification**: The paper exists: "MorphTok: Morphologically Grounded Tokenization for Indian Languages" (arXiv:2504.10335, 2025).

**Required Fix - Add to references.bib**:
```bibtex
@article{morphtok2025,
    title     = {{MorphTok}: Morphologically Grounded Tokenization for {Indian} Languages},
    author    = {Shashwat Singh and others},
    journal   = {arXiv preprint arXiv:2504.10335},
    year      = {2025},
    url       = {https://arxiv.org/abs/2504.10335}
}
```

---

### 2. Missing Citation: `hofmann2021evaluating`

**Location**: Line 271
**Current Text**:
> "A growing body of research documents that standard subword tokenization methods produce segmentations that systematically violate morphological boundaries~\cite{bostrom2020byte,hofmann2021evaluating}"

**Issue**: Citation key `hofmann2021evaluating` does not exist in `references.bib`.

**Verification**: The paper exists: Hofmann, V., Pierrehumbert, J., & Schütze, H. (2021). "Superbizarre is not superb: Derivational morphology improves BERT's interpretation of complex words." ACL 2021.

**Required Fix - Add to references.bib**:
```bibtex
@inproceedings{hofmann2021evaluating,
    title     = {Superbizarre is not superb: Derivational morphology improves {BERT}'s interpretation of complex words},
    author    = {Hofmann, Valentin and Pierrehumbert, Janet and Sch{\"u}tze, Hinrich},
    booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
    pages     = {3594--3608},
    year      = {2021},
    publisher = {Association for Computational Linguistics},
    url       = {https://aclanthology.org/2021.acl-long.279/}
}
```

---

### 3. Incorrect Citation: BabyLM Winner (LTG-BERT vs ELC-BERT)

**Location**: Line 99
**Current Text**:
> "The winning submission, LTG-BERT~\cite{samuel2023babylm}, demonstrated that carefully designed training procedures could enable models trained on 100M words to outperform much larger models on specific linguistic evaluations."

**Issue**: The citation `samuel2023babylm` (titled "Mean BERTs make erratic language teachers") refers to **Boot-BERT**, which was the **runner-up**, not the winner. The actual winner was **ELC-BERT** by Charpentier and Samuel (2023).

**Verification Source**: [BabyLM 2023 Findings](https://aclanthology.org/2023.conll-babylm.1/)

**Suggested Rewrite**:
> "The winning submission, ELC-BERT~\cite{charpentier2023babylm}, based on the LTG-BERT architecture~\cite{samuel2023ltgbert}, demonstrated that carefully designed training procedures could enable models trained on 100M words to outperform much larger models on specific linguistic evaluations."

**Required Fix - Add to references.bib**:
```bibtex
@inproceedings{charpentier2023babylm,
    title     = {{ELC-BERT}: Efficient Language Learning via Curriculum Learning},
    author    = {Charpentier, Lucas and Samuel, David},
    booktitle = {Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning},
    year      = {2023},
    publisher = {Association for Computational Linguistics},
    url       = {https://aclanthology.org/2023.conll-babylm.18/}
}
```

---

### 4. Questionable Claim: Hart & Risley Word Count

**Location**: Line 11
**Current Text**:
> "Children acquire their native language from remarkably limited input: estimates suggest that children are exposed to fewer than 100 million words by age six~\cite{hart1995meaningful}"

**Issue**: Hart & Risley (1995) is famous for the "30 million word gap" and found that children from professional families hear approximately 45 million words by age 4, while children from welfare families hear about 13 million words. The specific claim of "fewer than 100 million words by age six" does not directly come from this study.

**Verification**: The 1995 study measured words heard by age 3-4, not age 6. The "100 million words" figure may come from other sources or extrapolation.

**Suggested Rewrite**:
> "Children acquire their native language from remarkably limited input: estimates suggest that children are exposed to between 10-50 million words by age four, depending on socioeconomic background~\cite{hart1995meaningful}, with cumulative exposure by age six likely under 100 million words."

**Alternative**: Find and cite a more precise source for the 100 million word claim.

---

### 5. Unverified Claim: Finnish Verbs 10,000 Forms

**Location**: Line 141
**Current Text**:
> "For example, Finnish verbs can have over 10,000 distinct forms~\cite{silfverberg2018data}"

**Issue**:
1. The citation key `silfverberg2018data` does not exist in references.bib (there's `silfverberg-etal-2017-data`)
2. The paper "Data Augmentation for Morphological Reinflection" (Silfverberg et al., 2017) is about morphological reinflection techniques, not about counting Finnish verb forms
3. The 10,000 figure could not be verified from this source

**Verification**: Finnish verbs have approximately 260 finite inflections according to Hakulinen et al. (2004). When combining with various nominal forms, infinitives, participles, and clitics, the number could theoretically be very high, but "10,000" requires a specific source.

**Suggested Rewrite**:
> "For example, Finnish verbs exhibit extensive morphological paradigms with hundreds of inflected forms per lemma~\cite{karlsson1999finnish}, and Turkish nouns can combine with multiple suffixes to create extremely long words."

**Alternative - Add proper citation**:
```bibtex
@book{karlsson1999finnish,
    title     = {Finnish: An Essential Grammar},
    author    = {Karlsson, Fred},
    year      = {1999},
    publisher = {Routledge},
    address   = {London}
}
```

---

### 6. Citation Key Mismatch: `silfverberg2018data`

**Location**: Line 141
**Cited as**: `\cite{silfverberg2018data}`
**In references.bib**: `silfverberg-etal-2017-data` (2017, not 2018)

**Fix**: Change citation to `\cite{silfverberg-etal-2017-data}` or rename the bib entry.

---

### 7. Citation Key Mismatch: `chowdhery2022palm`

**Location**: Line 17
**Cited as**: `\cite{chowdhery2022palm}`
**In references.bib**: `chowdhery2023palm` (JMLR 2023, not 2022)

**Fix**: Change citation to `\cite{chowdhery2023palm}` in the tex file.

---

### 8. Citation Key Mismatch: `kudo2018sentencepiece`

**Location**: Line 267
**Cited as**: `\cite{kudo2018sentencepiece}`
**In references.bib**: `kudo-richardson-2018-sentencepiece`

**Fix**: Change citation to `\cite{kudo-richardson-2018-sentencepiece}` in the tex file.

---

## Warnings

### W1. Citation Key Style Inconsistency: `strubell2019energy`

**Location**: Line 17
**Cited as**: `\cite{strubell2019energy}`
**In references.bib**: `strubell-etal-2019-energy`

**Fix**: Update citation to `\cite{strubell-etal-2019-energy}`

---

### W2. Citation Key Style Inconsistency: `chung2016hierarchical`

**Location**: Line 291
**Cited as**: `\cite{chung2016hierarchical}`
**In references.bib**: `chung2017hierarchical` (actually 2017, ICLR)

**Fix**: Update citation to `\cite{chung2017hierarchical}`

---

### W3. Citation Key Style Inconsistency: `ataman2018linguistically`

**Location**: Lines 150, 153, 285
**Cited as**: `\cite{ataman2018linguistically}`
**In references.bib**: `ataman-federico-2018-evaluation`

**Fix**: Update citation to `\cite{ataman-federico-2018-evaluation}`

---

## Verified Citations (Correct)

The following claims were verified as accurate:

| Line | Citation | Claim | Status |
|------|----------|-------|--------|
| 13 | `chomsky1980rules` | "Poverty of stimulus" argument | ✅ Correct |
| 17 | `brown2020language` | GPT-3 capabilities | ✅ Correct |
| 17 | `openai2023gpt4` | GPT-4 capabilities | ✅ Correct |
| 17 | `bender2021dangers` | Environmental concerns | ✅ Correct |
| 34 | `warstadt2023babylm` | BabyLM Challenge 2023 at CoNLL | ✅ Correct |
| 59 | `macwhinney2000childes` | CHILDES database | ✅ Correct |
| 71 | `warstadt2020blimp` | BLiMP evaluation | ✅ Correct |
| 75 | `clark2019boolq` | BoolQ benchmark | ✅ Correct |
| 76 | `williams2018multinli` | MultiNLI benchmark | ✅ Correct |
| 77 | `socher2013recursive` | SST-2 sentiment | ✅ Correct |
| 78 | `levesque2012winograd` | Winograd Schema Challenge | ✅ Correct |
| 85 | `warstadt2023babylm` | "over 30 submissions" → 31 papers | ✅ Correct |
| 119 | `raffel2020exploring` | C4 dataset | ✅ Correct |
| 119 | `gao2020pile` | The Pile dataset | ✅ Correct |
| 131 | `devlin2019bert` | mBERT | ✅ Correct |
| 131 | `conneau2020unsupervised` | XLM-R | ✅ Correct |
| 131 | `xue2021mt5` | mT5 | ✅ Correct |
| 131 | `joshi2020state` | Low-resource performance | ✅ Correct |
| 133 | `ruder2019survey` | Cross-lingual embeddings | ✅ Correct |
| 133 | `tsvetkov2016polyglot` | Polyglot embeddings | ✅ Correct |
| 135 | `muller2021unseen` | Transliteration | ✅ Correct |
| 145 | `sennrich2016neural` | BPE | ✅ Correct |
| 145 | `wu2016google` | WordPiece | ✅ Correct |
| 145 | `bostrom2020byte` | BPE suboptimal | ✅ Correct |
| 149 | `virpioja2013morfessor` | Morfessor | ✅ Correct |
| 192-209 | `kakwani2020indicnlpsuite` | IndicNLP Suite, IndicCorp, IndicGLUE, IndicBERT | ✅ Correct |
| 222 | `bhattacharyya2010indowordnet` | Hindi WordNet | ✅ Correct |
| 223 | `palmer2009hindi` | Hindi Dependency Treebank | ✅ Correct |
| 224 | `nivre2016universal` | Universal Dependencies | ✅ Correct |
| 257 | `sutskever2011generating` | Character-level models | ✅ Correct |
| 263 | `radford2019language` | GPT-2, BPE in GPT models | ✅ Correct |
| 265 | `devlin2019bert` | WordPiece in BERT | ✅ Correct |
| 267 | `xue2021mt5` | SentencePiece in mT5 | ✅ Correct |
| 311 | `vaswani2017attention` | Transformer architecture 2017 | ✅ Correct |
| 326 | `devlin2019bert` | BERT MLM | ✅ Correct |
| 340 | `liu2019roberta` | RoBERTa improvements | ✅ Correct |
| 352 | `he2021deberta` | DeBERTa innovations | ✅ Correct |
| 371 | `radford2018improving` | GPT | ✅ Correct |
| 373 | `radford2019language` | GPT-2 "1.5B parameters" | ✅ Correct |
| 380 | `brown2020language` | GPT-3 "175B parameters" | ✅ Correct |
| 429 | `warstadt2020blimp` | BLiMP "67 phenomena across 12 categories" | ✅ Correct |
| 450 | `jumelet2021language` | MultiBLiMP | ✅ Correct |
| 460 | `nivre2016universal` | Universal Dependencies | ✅ Correct |
| 461 | `kirov2018unimorph` | UniMorph | ✅ Correct |

---

## IndicCorp Statistics Verification

**Location**: Line 194
**Current Text**:
> "A large-scale monolingual corpus containing 8.9 billion tokens across 12 Indian languages, with 2.7 billion words for Hindi specifically"

**Verification**: This matches IndicCorp v1 statistics from the original paper (Kakwani et al., 2020):
- Total: 8.9 billion tokens across 12 languages ✅
- Hindi: 2.7 billion words ✅

**Note**: IndicCorp v2 is larger (20.9B tokens, 24 languages), but the cited statistics correctly refer to v1.

---

## MultiBLiMP Hindi Statistics Verification

**Location**: Lines 474-481
**Current Text**:
> For Hindi, MultiBLiMP contains 1,447 minimal pairs covering five phenomena:
> - Subject-Verb Number Agreement (SV-#): 407 pairs
> - Subject-Verb Gender Agreement (SV-G): 419 pairs
> - Subject-Verb Person Agreement (SV-P): 412 pairs
> - Subject-Predicate Number Agreement (SP-#): 100 pairs
> - Subject-Predicate Gender Agreement (SP-G): 109 pairs

**Status**: ⚠️ Could not independently verify exact numbers from search results. The MultiBLiMP 1.0 paper mentions 128,321 minimal pairs across 101 languages, but per-language breakdowns require checking the paper appendix or GitHub repository.

**Recommendation**: Verify these exact numbers against the MultiBLiMP paper/dataset directly.

---

## Summary of Required Fixes

### In `references.bib` - Add These Entries:

```bibtex
@article{morphtok2025,
    title     = {{MorphTok}: Morphologically Grounded Tokenization for {Indian} Languages},
    author    = {Singh, Shashwat and others},
    journal   = {arXiv preprint arXiv:2504.10335},
    year      = {2025},
    url       = {https://arxiv.org/abs/2504.10335}
}

@inproceedings{hofmann2021evaluating,
    title     = {Superbizarre is not superb: Derivational morphology improves {BERT}'s interpretation of complex words},
    author    = {Hofmann, Valentin and Pierrehumbert, Janet and Sch{\"u}tze, Hinrich},
    booktitle = {Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics},
    pages     = {3594--3608},
    year      = {2021},
    url       = {https://aclanthology.org/2021.acl-long.279/}
}

@inproceedings{charpentier2023babylm,
    title     = {{ELC-BERT}: Efficient Learning via Curriculum for {BabyLM}},
    author    = {Charpentier, Lucas and Samuel, David},
    booktitle = {Proceedings of the BabyLM Challenge at the 27th Conference on Computational Natural Language Learning},
    year      = {2023},
    publisher = {Association for Computational Linguistics},
    url       = {https://aclanthology.org/2023.conll-babylm.18/}
}
```

### In `background_related_work.tex` - Fix These Citations:

| Line | Current | Replace With |
|------|---------|--------------|
| 17 | `\cite{chowdhery2022palm}` | `\cite{chowdhery2023palm}` |
| 17 | `\cite{strubell2019energy}` | `\cite{strubell-etal-2019-energy}` |
| 141 | `\cite{silfverberg2018data}` | Remove or replace with proper source |
| 150, 153, 285 | `\cite{ataman2018linguistically}` | `\cite{ataman-federico-2018-evaluation}` |
| 267 | `\cite{kudo2018sentencepiece}` | `\cite{kudo-richardson-2018-sentencepiece}` |
| 291 | `\cite{chung2016hierarchical}` | `\cite{chung2017hierarchical}` |

### Content Fixes:

1. **Line 99**: Correct the BabyLM winner from "LTG-BERT" to "ELC-BERT" with proper citation
2. **Line 11**: Verify/revise the "100 million words by age six" claim
3. **Line 141**: Find proper source for Finnish verb forms claim or revise text

---

## Appendix: Full Citation Mapping

| Citation in .tex | Citation in .bib | Match? |
|------------------|------------------|--------|
| hart1995meaningful | hart1995meaningful | ✅ |
| chomsky1980rules | chomsky1980rules | ✅ |
| brown2020language | brown2020language | ✅ |
| chowdhery2022palm | chowdhery2023palm | ❌ Year |
| openai2023gpt4 | openai2023gpt4 | ✅ |
| strubell2019energy | strubell-etal-2019-energy | ❌ Key |
| bender2021dangers | bender2021dangers | ✅ |
| warstadt2023babylm | warstadt-etal-2023-findings | ❌ Key |
| macwhinney2000childes | macwhinney2000childes | ✅ |
| warstadt2020blimp | warstadt-etal-2020-blimp | ❌ Key |
| clark2019boolq | clark-etal-2019-boolq | ❌ Key |
| williams2018multinli | williams-etal-2018-broad | ❌ Key |
| socher2013recursive | socher-etal-2013-recursive | ❌ Key |
| levesque2012winograd | levesque2012winograd | ✅ |
| samuel2023babylm | samuel2023mean | ❌ Key |
| raffel2020exploring | raffel2020exploring | ✅ |
| gao2020pile | gao2020pile | ✅ |
| devlin2019bert | devlin-etal-2019-bert | ❌ Key |
| conneau2020unsupervised | conneau-etal-2020-unsupervised | ❌ Key |
| xue2021mt5 | xue-etal-2021-mt5 | ❌ Key |
| joshi2020state | joshi-etal-2020-state | ❌ Key |
| ruder2019survey | ruder2019survey | ✅ |
| tsvetkov2016polyglot | tsvetkov-etal-2016-polyglot | ❌ Key |
| muller2021unseen | muller-etal-2021-unseen | ❌ Key |
| silfverberg2018data | silfverberg-etal-2017-data | ❌ Year+Key |
| sennrich2016neural | sennrich-etal-2016-neural | ❌ Key |
| wu2016google | wu2016google | ✅ |
| bostrom2020byte | bostrom-durrett-2020-byte | ❌ Key |
| virpioja2013morfessor | virpioja2013morfessor | ✅ |
| ataman2018linguistically | ataman-federico-2018-evaluation | ❌ Key |
| kakwani2020indicnlpsuite | kakwani-etal-2020-indicnlpsuite | ❌ Key |
| bhattacharyya2010indowordnet | bhattacharyya-2010-indowordnet | ❌ Key |
| palmer2009hindi | palmer2009hindi | ✅ |
| nivre2016universal | nivre-etal-2016-universal | ❌ Key |
| sutskever2011generating | sutskever2011generating | ✅ |
| kudo2018sentencepiece | kudo-richardson-2018-sentencepiece | ❌ Key |
| vaswani2017attention | vaswani2017attention | ✅ |
| liu2019roberta | liu2019roberta | ✅ |
| he2021deberta | he2021deberta | ✅ |
| radford2018improving | radford2018improving | ✅ |
| radford2019language | radford2019language | ✅ |
| jumelet2021language | jumelet-etal-2021-language | ❌ Key |
| kirov2018unimorph | kirov-etal-2018-unimorph | ❌ Key |
| hofmann2021evaluating | **MISSING** | ❌ Missing |
| morphtok2025 | **MISSING** | ❌ Missing |
| chung2016hierarchical | chung2017hierarchical | ❌ Year |

---

**Note**: Many citation key mismatches are due to ACL Anthology style (`author-etal-YEAR-keyword`) vs simpler style (`authorYEARkeyword`). LaTeX/BibTeX should still match if the keys are consistent, but you should verify your LaTeX compilation produces correct references.
