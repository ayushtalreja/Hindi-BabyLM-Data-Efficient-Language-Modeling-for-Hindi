# Introduction Chapter Citation Verification Report

**File Verified:** `introduction.tex`
**Reference File:** `references.bib`
**Verification Date:** 2026-01-20

---

## Executive Summary

| Status | Count |
|--------|-------|
| Citation Key Mismatches | 5 |
| Citation Key Correct | 1 |
| Factual Claims Verified | 8 |
| Factual Claims Incorrect | 0 |
| Minor Accuracy Issues | 1 |

---

## 1. Citation Key Verification

### 1.1 MISMATCHED Citation Keys (Require Correction)

| # | Citation Key in intro.tex | Correct Key in references.bib | Line |
|---|---------------------------|-------------------------------|------|
| 1 | `openai2024gpt4technicalreport` | `openai2023gpt4` | 8 |
| 2 | `chowdhery2022palmscalinglanguagemodeling` | `chowdhery2023palm` | 8 |
| 3 | `warstadt2023babylm` | `warstadt-etal-2023-findings` | 8, 15 |
| 4 | `strubell2019energy` | `strubell-etal-2019-energy` | 11 |
| 5 | `kakwani2020indicnlpsuite` | `kakwani-etal-2020-indicnlpsuite` | 37 |

### 1.2 CORRECT Citation Keys

| # | Citation Key | Status |
|---|--------------|--------|
| 1 | `bender2021dangers` | ✅ Matches references.bib |

### 1.3 Required Corrections in introduction.tex

**Line 8:**
```latex
% ORIGINAL (INCORRECT)
Systems like GPT-4~\cite{openai2024gpt4technicalreport}, and PaLM~\cite{chowdhery2022palmscalinglanguagemodeling} achieve human-level performance...

% CORRECTED
Systems like GPT-4~\cite{openai2023gpt4}, and PaLM~\cite{chowdhery2023palm} achieve human-level performance...
```

**Line 8 (continued):**
```latex
% ORIGINAL (INCORRECT)
...often comprising hundreds of billions or even trillions of tokens \cite{warstadt2023babylm}.

% CORRECTED
...often comprising hundreds of billions or even trillions of tokens \cite{warstadt-etal-2023-findings}.
```

**Line 11:**
```latex
% ORIGINAL (INCORRECT)
...computational cost and environmental impact of training ever-larger models~\cite{strubell2019energy,bender2021dangers}.

% CORRECTED
...computational cost and environmental impact of training ever-larger models~\cite{strubell-etal-2019-energy,bender2021dangers}.
```

**Line 15:**
```latex
% ORIGINAL (INCORRECT)
The BabyLM Challenge~\cite{warstadt2023babylm}, introduced in 2023...

% CORRECTED
The BabyLM Challenge~\cite{warstadt-etal-2023-findings}, introduced in 2023...
```

**Line 37:**
```latex
% ORIGINAL (INCORRECT)
...large-scale corpora (IndicCorp~\cite{kakwani2020indicnlpsuite})...

% CORRECTED
...large-scale corpora (IndicCorp~\cite{kakwani-etal-2020-indicnlpsuite})...
```

---

## 2. Factual Claim Verification

### 2.1 Claim: GPT-4 and PaLM achieve human-level performance

**Location:** Line 8
**Claim in Introduction:**
> "Systems like GPT-4 and PaLM achieve human-level performance on diverse tasks ranging from machine translation to complex reasoning."

**Source Verification:**

**GPT-4 Technical Report (OpenAI, 2023):**
> "While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers."

**PaLM (Chowdhery et al., 2023):**
> "PaLM 540B achieved breakthrough performance, outperforming the finetuned state-of-the-art on a suite of multi-step reasoning tasks, and outperforming average human performance on the recently released BIG-bench benchmark."

**Verdict:** ✅ **VERIFIED** - The claim is accurate. Both papers document human-level or superhuman performance on specific benchmarks.

---

### 2.2 Claim: Children learn from under 100 million words; models need 3-4 orders of magnitude more

**Location:** Line 8
**Claim in Introduction:**
> "Children typically master language from under 100 million words of input, yet state-of-the-art language models must be trained on datasets three to four orders of magnitude larger, often comprising hundreds of billions or even trillions of tokens."

**Source Verification (BabyLM Challenge, Warstadt et al., 2023):**
> "Children are incredibly data-efficient language learners compared to language models. Children are exposed to less than 100 million word tokens by age 13, while modern language models are typically trained on 3 or 4 orders-of-magnitude more data."

**Verdict:** ✅ **VERIFIED** - The claim directly matches the BabyLM Challenge findings paper.

---

### 2.3 Claim: Environmental and computational concerns

**Location:** Line 11
**Claim in Introduction:**
> "...speaks to broader concerns about the computational cost and environmental impact of training ever-larger models."

**Source Verification:**

**Strubell et al. (2019):**
> "These models are costly to train and develop, both financially, due to the cost of hardware and electricity or cloud compute time, and environmentally, due to the carbon footprint required to fuel modern tensor processing hardware."

**Bender et al. (2021):**
> "The paper considers environmental risks, echoing work outlining environmental and financial costs of deep learning systems, and encourages the research community to prioritize these impacts."

**Verdict:** ✅ **VERIFIED** - Both papers explicitly address computational cost and environmental concerns.

---

### 2.4 Claim: BabyLM Challenge tracks (10M, 100M words)

**Location:** Line 15
**Claim in Introduction:**
> "The BabyLM Challenge, introduced in 2023, operationalizes these questions by constraining language model training to developmentally plausible amounts of data. The challenge defines three tracks---10 million words ('strict-small'), 100 million words ('strict'), and 100 million words plus unlimited non-linguistic data ('loose')---approximating the linguistic input available to children at different developmental stages."

**Source Verification (BabyLM Challenge):**
> "The task has three tracks... The Strict track has a 100 million word limit, while the Strict-Small track has a 10 million word limit. The LOOSE track relaxes these restrictions [allowing unlimited non-text data]."

**Verdict:** ✅ **VERIFIED** - The track definitions are accurate.

---

### 2.5 Claim: Over 30 submissions in inaugural BabyLM challenge

**Location:** Line 17
**Claim in Introduction:**
> "Results from over 30 submissions in the inaugural challenge yielded important insights..."

**Source Verification:**
> "The challenge received 31 papers making a variety of contributions, ranging from designing novel architectures and tuning hyperparameters to employing curriculum learning and training teacher–student model pairs."

**Verdict:** ✅ **VERIFIED** - 31 papers were submitted, so "over 30" is accurate.

---

### 2.6 Claim: Hindi spoken by over 600 million people

**Location:** Line 23
**Claim in Introduction:**
> "Hindi, an Indo-Aryan language spoken by over 600 million people, provides an ideal testbed..."

**Source Verification (Multiple Sources):**
- Ethnologue: "approximately 600 million people who speak Hindi as either their first or second language"
- News on AIR: "Hindi is 3rd most spoken language in the world with 615 million speakers"
- Various sources cite 600-615 million total speakers

**Verdict:** ✅ **VERIFIED** - The "over 600 million" claim is well-supported by linguistic census data.

---

### 2.7 Claim: IndicCorp and IndicNLP Suite resources

**Location:** Line 37
**Claim in Introduction:**
> "Despite substantial progress in Hindi NLP---including large-scale corpora (IndicCorp), pretrained models (IndicBERT, IndicBART), and comprehensive benchmarks (IndicGLUE, MultiBLiMP)---no prior work has systematically investigated..."

**Source Verification (Kakwani et al., 2020):**
> "IndicNLPSuite consists of a large-scale monolingual corpora IndicCorp, a pre-trained language model IndicBERT, pretrained word-embeddings IndicFT and a general natural language understanding benchmark IndicGLUE."
> "IndicCorp is one of the largest publicly-available corpora for Indian languages."

**Verdict:** ✅ **VERIFIED** - All mentioned resources (IndicCorp, IndicBERT, IndicGLUE) are correctly attributed to the IndicNLP Suite.

**Note:** IndicBART and MultiBLiMP are not from this paper but exist as separate resources. Consider adding their respective citations if needed.

---

### 2.8 Claim: BabyLM focused exclusively on English

**Location:** Line 19
**Claim in Introduction:**
> "However, the BabyLM challenge has focused exclusively on English, leaving critical questions unanswered..."

**Source Verification:**
The 2023 BabyLM Challenge dataset contained only English text, and all evaluations (BLiMP, SuperGLUE subsets) were English-only.

**Verdict:** ✅ **VERIFIED** - The original BabyLM Challenge was indeed English-only.

---

## 3. Potential Issues and Recommendations

### 3.1 Minor Issue: IndicBART and MultiBLiMP Citations

**Issue:** Line 37 mentions "IndicBART" and "MultiBLiMP" alongside IndicCorp but cites only `kakwani-etal-2020-indicnlpsuite`, which does not include these resources.

**Recommendation:** Consider adding citations for:
- **IndicBART**: Check for the appropriate citation (AI4Bharat/IndicTrans papers)
- **MultiBLiMP**: The multilingual extension of BLiMP (not in current references.bib)

### 3.2 Year Inconsistency in Citation Keys

The references.bib uses the ACL Anthology citation key format (`author-etal-YEAR-keyword`) for most entries, but introduction.tex uses a different format. Consider standardizing:

| Current Format | ACL Anthology Format |
|----------------|---------------------|
| `openai2024gpt4technicalreport` | `openai2023gpt4` |
| `warstadt2023babylm` | `warstadt-etal-2023-findings` |

---

## 4. Summary of Required Changes

### 4.1 Critical (Must Fix)

Replace the following citation keys in `introduction.tex`:

```
openai2024gpt4technicalreport → openai2023gpt4
chowdhery2022palmscalinglanguagemodeling → chowdhery2023palm
warstadt2023babylm → warstadt-etal-2023-findings
strubell2019energy → strubell-etal-2019-energy
kakwani2020indicnlpsuite → kakwani-etal-2020-indicnlpsuite
```

### 4.2 Optional (Enhancement)

1. Add citations for IndicBART and MultiBLiMP if they are mentioned as separate resources
2. Consider whether to add a citation for Hindi speaker statistics (e.g., Ethnologue or Census of India)

---

## 5. Verification Sources

- [GPT-4 Technical Report - arXiv](https://arxiv.org/abs/2303.08774)
- [PaLM: Scaling Language Modeling - JMLR](https://jmlr.org/papers/v24/22-1144.html)
- [BabyLM Challenge 2023](https://babylm.github.io/archive_2023.html)
- [BabyLM Findings Paper - ACL Anthology](https://aclanthology.org/2023.conll-babylm.1/)
- [IndicNLPSuite - ACL Anthology](https://aclanthology.org/2020.findings-emnlp.445/)
- [Energy and Policy Considerations - ACL Anthology](https://aclanthology.org/P19-1355/)
- [Stochastic Parrots - ACM FAccT](https://dl.acm.org/doi/10.1145/3442188.3445922)

---

**Report Generated:** 2026-01-20
**Verification Status:** All factual claims verified; citation key corrections required
