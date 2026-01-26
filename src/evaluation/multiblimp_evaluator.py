"""
MultiBLiMP (Multilingual BLiMP) Evaluator for Hindi

This module implements comprehensive evaluation of syntactic phenomena in Hindi
using minimal pair testing methodology. Each test consists of a grammatical
sentence paired with an ungrammatical variant differing in a single linguistic
feature.

Phenomena Tested (from jumelet/multiblimp dataset):
- SV-#: Subject-Verb Number Agreement (407 pairs)
- SV-G: Subject-Verb Gender Agreement (419 pairs)
- SV-P: Subject-Verb Person Agreement (412 pairs)
- SP-#: Subject-Predicate Number Agreement (100 pairs)
- SP-G: Subject-Predicate Gender Agreement (109 pairs)

Total: 1447 minimal pairs across 5 phenomena

Reference: https://github.com/alexwarstadt/blimp
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset, Dataset
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class MultiBLiMPEvaluator:
    """
    Comprehensive evaluator for Hindi syntactic phenomena using minimal pairs

    Features:
    - 5 linguistic phenomena tested (agreement phenomena)
    - Perplexity-based evaluation
    - Comprehensive minimal pair database (1447 pairs total)
    - Statistical analysis
    - Per-phenomenon metrics
    - Overall syntactic competence score
    """

    def __init__(self, model, tokenizer, config: Optional[Dict] = None):
        """
        Initialize MultiBLiMP evaluator

        Args:
            model: Language model to evaluate (will unwrap if classification-wrapped)
            tokenizer: Tokenizer for the model
            config: Optional configuration dictionary

        Note:
            MultiBLiMP evaluation requires language modeling capabilities (per-token logits).
            If you pass a classification-wrapped model, it will be automatically unwrapped
            to access the base language model. If unwrapping fails, you'll need to pass
            the base LM directly.
        """
        # Detect and unwrap classification models
        if self._is_classification_wrapper(model):
            logger.warning(
                f"Detected classification-wrapped model ({model.__class__.__name__}). "
                f"MultiBLiMP requires language modeling capabilities. Attempting to unwrap..."
            )
            try:
                model = self._unwrap_classification_model(model)
                logger.info(f"Successfully unwrapped to {model.__class__.__name__}")
            except ValueError as e:
                logger.error(str(e))
                raise

        self.model = model

        # Diagnostic logging for model structure
        logger.info(f"Final model type for MultiBLiMP: {model.__class__.__name__}")
        logger.info(f"Model module: {model.__class__.__module__}")
        model_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
        logger.info(f"Model has {len(model_attrs)} public attributes")
        logger.debug(f"Model attributes: {model_attrs}")

        self.tokenizer = tokenizer
        self.config = config or {}

        # Device setup
        self.device = next(model.parameters()).device
        logger.info(f"MultiBLiMP evaluator initialized on device: {self.device}")

        # Detect model type (MLM vs CLM)
        # MLM models (DeBERTa, BERT) need pseudo-log-likelihood evaluation
        # CLM models (GPT) use standard next-token prediction
        self.is_mlm = self._detect_is_mlm_model()
        logger.info(f"Model type: {'MLM (bidirectional)' if self.is_mlm else 'CLM (autoregressive)'}")
        if self.is_mlm:
            logger.info("Will use pseudo-log-likelihood for perplexity computation")

        # Get config parameters
        multiblimp_config = self.config.get('evaluation', {}).get('benchmarks', {}).get('multiblimp', {})
        self.max_examples_per_phenomenon = multiblimp_config.get('n_examples_per_phenomenon', None)

        if self.max_examples_per_phenomenon:
            logger.info(f"Will limit to {self.max_examples_per_phenomenon} examples per phenomenon (from config)")

        # Phenomena to test (actual names from jumelet/multiblimp dataset)
        self.phenomena = [
            'SV-#',   # Subject-Verb Number Agreement
            'SV-G',   # Subject-Verb Gender Agreement
            'SV-P',   # Subject-Verb Person Agreement
            'SP-#',   # Subject-Predicate Number Agreement
            'SP-G',   # Subject-Predicate Gender Agreement
        ]

        # Load or create minimal pairs
        self.minimal_pairs = self._initialize_minimal_pairs()

        # Log comprehensive test coverage statistics
        total_pairs = sum(len(pairs) for pairs in self.minimal_pairs.values())
        logger.info(f"MultiBLiMP test coverage: {total_pairs} minimal pairs across {len(self.minimal_pairs)} phenomena")

        # Show per-phenomenon breakdown
        if total_pairs > 100:  # External dataset loaded (detailed breakdown)
            logger.info("Per-phenomenon test coverage:")
            for phenomenon in sorted(self.minimal_pairs.keys()):
                count = len(self.minimal_pairs[phenomenon])
                logger.info(f"  {phenomenon}: {count} pairs")
        else:  # Built-in dataset (summary only)
            logger.info(f"Using built-in test suite ({total_pairs} pairs total)")

        # Validate model outputs correct structure - fail fast if incompatible
        # Can be disabled for testing with skip_init_validation=True in config
        if not self.config.get('skip_init_validation', False):
            logger.info("Validating model output structure...")
            try:
                # Create minimal test input
                test_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(self.device)

                # Run forward pass
                with torch.no_grad():
                    test_output = self.model(input_ids=test_ids)

                # Extract and validate logits
                test_logits = self._extract_logits(test_output, "initialization")

                if test_logits.dim() != 3:
                    raise ValueError(
                        f"Model outputs {test_logits.dim()}D logits (shape: {test_logits.shape}), "
                        f"expected 3D [batch, seq_len, vocab_size]"
                    )

                # Verify reasonable vocabulary size
                vocab_dim = test_logits.size(-1)
                if vocab_dim < 1000:
                    logger.warning(
                        f"Vocabulary size is {vocab_dim}, which seems small for a language model. "
                        f"Expected 10k-50k. This might indicate a classification model."
                    )

                logger.info(f"✓ Model validation passed!")
                logger.info(f"  Output shape: {test_logits.shape}")
                logger.info(f"  Vocabulary size: {vocab_dim}")

            except Exception as e:
                error_msg = (
                    f"\n{'='*70}\n"
                    f"MULTIBLIMP INITIALIZATION ERROR\n"
                    f"{'='*70}\n"
                    f"The model is not compatible with MultiBLiMP evaluation.\n\n"
                    f"Model class: {self.model.__class__.__name__}\n"
                    f"Error: {str(e)}\n\n"
                    f"MultiBLiMP requires a LANGUAGE MODEL that outputs per-token predictions:\n"
                    f"  Expected: [batch, sequence_length, vocabulary_size] (3D)\n"
                    f"  Example: [1, 128, 32000] for a model with 32k vocab\n\n"
                    f"If you're using a classification-wrapped model, ensure it gets unwrapped.\n"
                    f"{'='*70}\n"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
        else:
            logger.debug("Skipping initialization validation (skip_init_validation=True in config)")

    def _is_classification_wrapper(self, model) -> bool:
        """
        Detect if model is a classification wrapper.

        Returns:
            True if model is wrapped for classification
        """
        # Check for classification wrapper classes
        wrapper_classes = [
            'GPTForSequenceClassification',
            'DeBERTaForSequenceClassification',
            'SequenceClassification'  # Generic pattern
        ]

        model_class_name = model.__class__.__name__
        return any(wrapper in model_class_name for wrapper in wrapper_classes)

    def _unwrap_classification_model(self, model):
        """
        Extract base language model from classification wrapper.

        Args:
            model: Classification-wrapped model

        Returns:
            Unwrapped language model

        Raises:
            ValueError: If unwrapping fails
        """
        # Try to access lm_model attribute (our wrapper pattern)
        if hasattr(model, 'lm_model'):
            logger.info("Detected classification wrapper, extracting base LM...")
            return model.lm_model

        # Try to access base_model (HuggingFace pattern)
        if hasattr(model, 'base_model'):
            logger.info("Detected HuggingFace classification wrapper, extracting base model...")
            return model.base_model

        # Try to access model attribute (HuggingFace GPT2LMHeadModel pattern)
        if hasattr(model, 'model'):
            logger.info("Extracting base model from 'model' attribute...")
            return model.model

        raise ValueError(
            f"Model appears to be a classification wrapper ({model.__class__.__name__}) "
            f"but could not extract base language model. MultiBLiMP requires a language "
            f"model that outputs per-token logits [batch, seq_len, vocab_size]. "
            f"Available attributes: {list(model.__dict__.keys())}"
        )

    def _detect_is_mlm_model(self) -> bool:
        """
        Detect if the model is an MLM (Masked Language Model) like DeBERTa/BERT
        or a CLM (Causal Language Model) like GPT.

        MLM models require pseudo-log-likelihood evaluation since they can't
        do next-token prediction (they're bidirectional).

        Returns:
            True if MLM model, False if CLM model
        """
        model_class_name = self.model.__class__.__name__.lower()

        # Check for MLM indicators
        mlm_indicators = ['deberta', 'bert', 'albert', 'roberta', 'xlm', 'electra', 'maskedlm']
        if any(indicator in model_class_name for indicator in mlm_indicators):
            return True

        # Check for CLM indicators
        clm_indicators = ['gpt', 'causal', 'lmhead']
        if any(indicator in model_class_name for indicator in clm_indicators):
            return False

        # Check config model_type if available
        config_model_type = self.config.get('model', {}).get('type',
                           self.config.get('model_type', '')).lower()
        if config_model_type in ['deberta', 'bert', 'albert', 'roberta']:
            return True
        if config_model_type in ['gpt', 'gpt2', 'gpt-2']:
            return False

        # Default: assume CLM for safety (standard BLiMP behavior)
        logger.warning(f"Could not determine model type from {model_class_name}, assuming CLM")
        return False

    def _compute_pseudo_log_likelihood(self, sentence: str) -> float:
        """
        Compute pseudo-log-likelihood (PLL) for a sentence using MLM.

        For bidirectional models (BERT, DeBERTa), we can't use standard perplexity.
        Instead, we mask each token one at a time and sum the log probabilities.

        PLL(s) = sum over all tokens t: log P(t | context_without_t)

        Args:
            sentence: Input sentence

        Returns:
            Pseudo-log-likelihood (negative, lower = more probable)
        """
        # Tokenize sentence
        inputs = self._tokenize_sentence(sentence)
        input_ids = inputs['input_ids']  # [1, seq_len]
        attention_mask = inputs['attention_mask']  # [1, seq_len]

        seq_len = input_ids.size(1)

        # Get mask token id
        if self.tokenizer.mask_token_id is None:
            # Fallback: use a common mask token id
            logger.warning("Tokenizer has no mask_token_id, attempting to find [MASK] token")
            mask_token_id = self.tokenizer.convert_tokens_to_ids('[MASK]')
            if mask_token_id == self.tokenizer.unk_token_id:
                raise ValueError("Cannot compute PLL: tokenizer has no [MASK] token")
        else:
            mask_token_id = self.tokenizer.mask_token_id

        total_log_prob = 0.0
        num_tokens = 0

        # Mask each token one at a time and compute log probability
        for pos in range(seq_len):
            # Skip special tokens (padding, etc.)
            token_id = input_ids[0, pos].item()
            if token_id == self.tokenizer.pad_token_id:
                continue
            if token_id == self.tokenizer.cls_token_id:
                continue
            if token_id == self.tokenizer.sep_token_id:
                continue
            if token_id == self.tokenizer.bos_token_id:
                continue
            if token_id == self.tokenizer.eos_token_id:
                continue

            # Create masked input
            masked_input_ids = input_ids.clone()
            masked_input_ids[0, pos] = mask_token_id

            # Forward pass
            outputs = self.model(
                input_ids=masked_input_ids,
                attention_mask=attention_mask
            )

            # Extract logits
            logits = self._extract_logits(outputs, "pll")  # [1, seq_len, vocab_size]

            # Get log probabilities for the masked position
            log_probs = torch.log_softmax(logits[0, pos, :], dim=-1)

            # Get log probability of the original token
            log_prob = log_probs[token_id].item()
            total_log_prob += log_prob
            num_tokens += 1

        # Return negative log likelihood (lower = more probable)
        # We negate because we want lower loss = better
        return -total_log_prob if num_tokens > 0 else float('inf')

    def _evaluate_with_pseudo_likelihood(self, good_sentence: str, bad_sentence: str) -> Tuple[bool, float, float, float]:
        """
        Evaluate a minimal pair using pseudo-log-likelihood for MLM models.

        Args:
            good_sentence: Grammatical sentence
            bad_sentence: Ungrammatical sentence

        Returns:
            Tuple of (is_correct, good_pll, bad_pll, pll_difference)
        """
        try:
            # Compute PLL for both sentences
            good_pll = self._compute_pseudo_log_likelihood(good_sentence)
            bad_pll = self._compute_pseudo_log_likelihood(bad_sentence)

            # Model should assign lower PLL (higher probability) to grammatical sentence
            is_correct = good_pll < bad_pll
            pll_difference = bad_pll - good_pll

            return is_correct, good_pll, bad_pll, pll_difference

        except Exception as e:
            logger.warning(f"Error in PLL computation: {e}")
            return False, float('inf'), float('inf'), 0.0

    def evaluate_all_phenomena(self) -> Dict[str, Dict]:
        """
        Evaluate model on all syntactic phenomena

        Returns:
            Dictionary mapping phenomenon names to results
        """
        logger.info("Starting MultiBLiMP evaluation on all phenomena...")
        results = {}

        for phenomenon in self.phenomena:
            if phenomenon not in self.minimal_pairs:
                logger.warning(f"No minimal pairs found for {phenomenon}")
                continue

            logger.info(f"\nEvaluating {phenomenon}...")

            try:
                phenomenon_results = self.evaluate_phenomenon(
                    phenomenon,
                    self.minimal_pairs[phenomenon]
                )
                results[phenomenon] = phenomenon_results

                # Log results
                logger.info(f"{phenomenon} Results:")
                logger.info(f"  Accuracy: {phenomenon_results['accuracy']:.4f}")
                logger.info(f"  Correct: {phenomenon_results['correct']}/{phenomenon_results['total']}")

            except Exception as e:
                logger.error(f"Error evaluating {phenomenon}: {str(e)}")
                results[phenomenon] = {'error': str(e), 'status': 'failed'}

        # Compute overall statistics
        results['overall'] = self._compute_overall_metrics(results)

        logger.info("\n" + "="*60)
        logger.info("MultiBLiMP Evaluation Complete")
        logger.info(f"Overall Accuracy: {results['overall']['average_accuracy']:.4f}")
        logger.info("="*60)

        return results

    def evaluate_phenomenon(self, phenomenon: str, pairs: List[Dict]) -> Dict:
        """
        Evaluate all minimal pairs for a specific phenomenon

        Args:
            phenomenon: Name of the phenomenon
            pairs: List of minimal pair dictionaries

        Returns:
            Dictionary with evaluation metrics
        """
        correct_predictions = 0
        total_pairs = len(pairs)

        # Track detailed results
        pair_results = []
        loss_differences = []
        good_losses = []
        bad_losses = []
        length_diffs = []  # Track character length differences (good - bad)

        self.model.eval()

        with torch.no_grad():
            for idx, pair in enumerate(tqdm(pairs, desc=f"Evaluating {phenomenon}")):
                try:
                    # Evaluate minimal pair
                    is_correct, good_loss, bad_loss, loss_diff = self._evaluate_minimal_pair_detailed(
                        pair['good'],
                        pair['bad']
                    )

                    if is_correct:
                        correct_predictions += 1

                    loss_differences.append(loss_diff)
                    good_losses.append(good_loss)
                    bad_losses.append(bad_loss)
                    length_diffs.append(len(pair['good']) - len(pair['bad']))

                    pair_results.append({
                        'good': pair['good'],
                        'bad': pair['bad'],
                        'correct': is_correct,
                        'good_loss': good_loss,
                        'bad_loss': bad_loss,
                        'loss_difference': loss_diff
                    })

                except ValueError as e:
                    # Dimension validation error - this is fatal
                    logger.error(
                        f"Dimension validation failed on {phenomenon} pair {idx+1}/{total_pairs}: {e}"
                    )
                    raise  # Re-raise to stop evaluation

                except Exception as e:
                    # Other errors - log and continue
                    logger.warning(
                        f"Failed to evaluate {phenomenon} pair {idx+1}/{total_pairs}: {e}"
                    )
                    # Don't count as correct, but continue evaluation

        # Compute metrics
        accuracy = correct_predictions / total_pairs if total_pairs > 0 else 0

        # Compute diagnostic metrics
        below_chance = accuracy < 0.5
        prefers_ungrammatical = np.mean(loss_differences) < 0 if loss_differences else False

        results = {
            'phenomenon': phenomenon,
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': total_pairs,
            'mean_loss_difference': np.mean(loss_differences) if loss_differences else 0,
            'std_loss_difference': np.std(loss_differences) if loss_differences else 0,
            # Diagnostic metrics for understanding below-chance performance
            'diagnostics': {
                'below_chance': below_chance,
                'prefers_ungrammatical': prefers_ungrammatical,
                'mean_good_loss': np.mean(good_losses) if good_losses else 0,
                'mean_bad_loss': np.mean(bad_losses) if bad_losses else 0,
                'mean_length_diff': np.mean(length_diffs) if length_diffs else 0,
                'pairs_good_longer': sum(1 for d in length_diffs if d > 0),
                'pairs_same_length': sum(1 for d in length_diffs if d == 0),
                'pairs_bad_longer': sum(1 for d in length_diffs if d < 0),
            },
            'pair_results': pair_results if self.config.get('save_pair_results', False) else None
        }

        # Log warning for below-chance performance
        if below_chance:
            logger.warning(
                f"{phenomenon}: Below-chance accuracy ({accuracy:.1%}). "
                f"Model {'prefers ungrammatical' if prefers_ungrammatical else 'shows no clear preference'}. "
                f"Mean loss diff: {results['mean_loss_difference']:.4f}"
            )

        return results

    def _evaluate_minimal_pair_detailed(self, good_sentence: str, bad_sentence: str) -> Tuple[bool, float, float, float]:
        """
        Evaluate a single minimal pair with detailed metrics

        For MLM models (DeBERTa, BERT), uses pseudo-log-likelihood.
        For CLM models (GPT), uses standard next-token prediction.

        Args:
            good_sentence: Grammatical sentence
            bad_sentence: Ungrammatical sentence

        Returns:
            Tuple of (is_correct, good_loss, bad_loss, loss_difference)
        """
        # Route to appropriate evaluation method based on model type
        if self.is_mlm:
            return self._evaluate_with_pseudo_likelihood(good_sentence, bad_sentence)

        # CLM evaluation (standard next-token prediction) follows below
        try:
            # Tokenize both sentences
            good_inputs = self._tokenize_sentence(good_sentence)
            bad_inputs = self._tokenize_sentence(bad_sentence)

            # Get model outputs WITHOUT labels parameter
            good_outputs = self.model(**good_inputs)
            bad_outputs = self.model(**bad_inputs)

            # Extract logits with better error handling
            good_logits = self._extract_logits(good_outputs, "good")
            bad_logits = self._extract_logits(bad_outputs, "bad")

            # CRITICAL: Validate dimensions IMMEDIATELY - BEFORE any other operations
            # This prevents cryptic "Dimension out of range" errors from masking the real issue
            if good_logits.dim() != 3:
                raise ValueError(
                    f"MultiBLiMP ERROR: Model outputs {good_logits.dim()}D logits, expected 3D [batch, seq_len, vocab_size].\n"
                    f"  Got shape: {good_logits.shape}\n"
                    f"  Model class: {self.model.__class__.__name__}\n"
                    f"  Model module: {self.model.__class__.__module__}\n"
                    f"  Output type: {type(good_outputs)}\n\n"
                    f"This means the model is outputting per-sequence predictions (classification) "
                    f"instead of per-token predictions (language modeling).\n"
                    f"MultiBLiMP requires a language model, not a classification model."
                )

            if bad_logits.dim() != 3:
                raise ValueError(
                    f"MultiBLiMP ERROR: Model outputs {bad_logits.dim()}D logits for bad sentence.\n"
                    f"  Shape: {bad_logits.shape}, Model: {self.model.__class__.__name__}"
                )

            # Verify reasonable vocabulary size
            vocab_size = good_logits.size(-1)
            if vocab_size < 100:
                logger.warning(
                    f"Small vocabulary size ({vocab_size}). Expected 10k-50k for LM. "
                    f"This might indicate a classification model."
                )

            # Verify sequence length is sufficient for shifting
            seq_len = good_logits.size(1)
            if seq_len < 2:
                logger.warning(
                    f"Sequence length ({seq_len}) too short for next-token prediction. "
                    f"Need at least 2 tokens. Skipping this pair."
                )
                return False, float('inf'), float('inf'), 0.0

            # Compute loss for good sentence
            # Shift logits and labels for next-token prediction
            good_shift_logits = good_logits[:, :-1, :].contiguous()
            good_shift_labels = good_inputs['input_ids'][:, 1:].contiguous()

            # Compute cross-entropy loss with per-token averaging
            # Using 'mean' reduction normalizes by sequence length, ensuring fair comparison
            # for tokenizers with different granularity (character-level vs subword)
            loss_fct = torch.nn.CrossEntropyLoss(reduction='mean')
            good_loss = loss_fct(
                good_shift_logits.view(-1, good_shift_logits.size(-1)),
                good_shift_labels.view(-1)
            ).item()

            # Compute loss for bad sentence
            bad_shift_logits = bad_logits[:, :-1, :].contiguous()
            bad_shift_labels = bad_inputs['input_ids'][:, 1:].contiguous()

            bad_loss = loss_fct(
                bad_shift_logits.view(-1, bad_shift_logits.size(-1)),
                bad_shift_labels.view(-1)
            ).item()

            # Model should prefer (assign lower loss to) grammatical sentence
            is_correct = good_loss < bad_loss
            loss_difference = bad_loss - good_loss

            return is_correct, good_loss, bad_loss, loss_difference

        except ValueError as e:
            # Re-raise ValueError with context (dimension validation errors)
            logger.error(f"Validation error in minimal pair evaluation: {e}")
            raise
        except Exception as e:
            # Log other errors but return failure gracefully
            logger.warning(f"Error evaluating pair: {e}")
            return False, float('inf'), float('inf'), 0.0

    def _extract_logits(self, model_outputs, sentence_type: str = "") -> torch.Tensor:
        """
        Safely extract logits from model outputs with comprehensive error handling.

        Args:
            model_outputs: Model output object
            sentence_type: Descriptor for error messages ("good"/"bad")

        Returns:
            Logits tensor [batch, seq_len, vocab_size]

        Raises:
            ValueError: If logits cannot be extracted or have wrong dimensions
        """
        logits = None

        # Try to extract logits attribute
        if hasattr(model_outputs, 'logits'):
            logits = model_outputs.logits
        # Try tuple/list indexing
        elif isinstance(model_outputs, (tuple, list)) and len(model_outputs) > 0:
            logits = model_outputs[0]
        # Try dict-like access
        elif isinstance(model_outputs, dict) and 'logits' in model_outputs:
            logits = model_outputs['logits']
        else:
            raise ValueError(
                f"Cannot extract logits from {sentence_type} sentence output. "
                f"Output type: {type(model_outputs)}. "
                f"Available attributes: {dir(model_outputs) if hasattr(model_outputs, '__dir__') else 'N/A'}"
            )

        # Verify we got a tensor
        if not isinstance(logits, torch.Tensor):
            raise ValueError(
                f"Extracted logits are not a tensor for {sentence_type} sentence. "
                f"Got type: {type(logits)}"
            )

        # Early dimension check with helpful error message
        if logits.dim() < 2:
            raise ValueError(
                f"Logits for {sentence_type} sentence have {logits.dim()} dimensions, "
                f"expected at least 2D tensor. Shape: {logits.shape}"
            )

        return logits

    def _tokenize_sentence(self, sentence: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize a sentence and move to device

        Args:
            sentence: Input sentence

        Returns:
            Dictionary with tokenized inputs
        """
        try:
            # Try HuggingFace tokenizer interface
            encoded = self.tokenizer(
                sentence,
                return_tensors='pt',
                padding=False,
                truncation=True,
                max_length=128
            )
        except:
            # Fallback to simple encoding
            tokens = self.tokenizer.encode(sentence)
            encoded = {
                'input_ids': torch.tensor([tokens]),
                'attention_mask': torch.ones(len(tokens))
            }

        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        return encoded

    def _compute_overall_metrics(self, results: Dict[str, Dict]) -> Dict:
        """
        Compute overall statistics across all phenomena

        Args:
            results: Dictionary of per-phenomenon results

        Returns:
            Dictionary with overall metrics
        """
        accuracies = []
        loss_diffs = []
        total_correct = 0
        total_pairs = 0

        for phenomenon, phenomenon_results in results.items():
            if phenomenon == 'overall':
                continue

            if 'accuracy' in phenomenon_results:
                accuracies.append(phenomenon_results['accuracy'])
                total_correct += phenomenon_results.get('correct', 0)
                total_pairs += phenomenon_results.get('total', 0)

                if 'mean_loss_difference' in phenomenon_results:
                    loss_diffs.append(phenomenon_results['mean_loss_difference'])

        # Count below-chance phenomena
        below_chance_phenomena = [
            phenomenon for phenomenon in self.phenomena
            if phenomenon in results
            and 'diagnostics' in results[phenomenon]
            and results[phenomenon]['diagnostics'].get('below_chance', False)
        ]

        overall = {
            'average_accuracy': np.mean(accuracies) if accuracies else 0.0,
            'std_accuracy': np.std(accuracies) if accuracies else 0.0,
            'min_accuracy': np.min(accuracies) if accuracies else 0.0,
            'max_accuracy': np.max(accuracies) if accuracies else 0.0,
            'total_correct': total_correct,
            'total_pairs': total_pairs,
            'overall_accuracy': total_correct / total_pairs if total_pairs > 0 else 0.0,
            'phenomena_evaluated': len(accuracies),
            'mean_loss_difference': np.mean(loss_diffs) if loss_diffs else 0.0,
            'accuracies_by_phenomenon': {
                phenomenon: results[phenomenon].get('accuracy', 0)
                for phenomenon in self.phenomena
                if phenomenon in results and 'accuracy' in results[phenomenon]
            },
            # Overall diagnostics
            'below_chance_count': len(below_chance_phenomena),
            'below_chance_phenomena': below_chance_phenomena,
            'overall_below_chance': (total_correct / total_pairs if total_pairs > 0 else 0.0) < 0.5,
        }

        # Log warning if overall performance is below chance
        if overall['overall_below_chance']:
            logger.warning(
                f"CRITICAL: Overall MultiBLiMP accuracy ({overall['overall_accuracy']:.1%}) is below chance (50%). "
                f"{len(below_chance_phenomena)}/{len(accuracies)} phenomena show below-chance performance. "
                f"This may indicate: (1) tokenization issues affecting morphological patterns, "
                f"(2) insufficient training data for learning agreement, or "
                f"(3) model architecture limitations for capturing syntactic dependencies."
            )

        return overall

    def _initialize_minimal_pairs(self) -> Dict[str, List[Dict]]:
        """
        Initialize minimal pairs database

        Returns:
            Dictionary mapping phenomena to lists of minimal pairs

        Raises:
            RuntimeError: If the MultiBLiMP dataset cannot be loaded
        """
        # Load from HuggingFace
        minimal_pairs = self._load_multiblimp_dataset()

        if not minimal_pairs:
            error_msg = (
                "Failed to load MultiBLiMP dataset from HuggingFace (jumelet/multiblimp). "
                "This dataset is required for evaluation. Please ensure you have internet "
                "connectivity and the datasets library is installed."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Apply limiting if configured
        if self.max_examples_per_phenomenon:
            original_total = sum(len(pairs) for pairs in minimal_pairs.values())
            limited_pairs = {}
            for phenomenon, pairs in minimal_pairs.items():
                if len(pairs) > self.max_examples_per_phenomenon:
                    limited_pairs[phenomenon] = pairs[:self.max_examples_per_phenomenon]
                    logger.debug(f"Limited {phenomenon} from {len(pairs)} to {self.max_examples_per_phenomenon} pairs")
                else:
                    limited_pairs[phenomenon] = pairs

            new_total = sum(len(pairs) for pairs in limited_pairs.values())
            if new_total < original_total:
                logger.info(f"Applied limiting: reduced from {original_total} to {new_total} total pairs")
            minimal_pairs = limited_pairs

        return minimal_pairs

    def _load_multiblimp_dataset(self) -> Optional[Dict]:
        """
        Load MultiBLiMP dataset from HuggingFace

        Returns:
            Dataset or None if not available
        """
        try:
            logger.info("Attempting to load MultiBLiMP dataset from HuggingFace (jumelet/multiblimp)...")
            dataset = load_dataset('jumelet/multiblimp', 'hin', split='train')

            logger.info(f"Successfully loaded external dataset with {len(dataset)} examples")

            # Convert to our format
            minimal_pairs = defaultdict(list)
            for example in dataset:
                phenomenon = example.get('phenomenon', 'unknown')
                minimal_pairs[phenomenon].append({
                    'good': example['sen'],  # Grammatical sentence
                    'bad': example['wrong_sen'],  # Ungrammatical sentence
                    'phenomenon': phenomenon
                })

            # Log statistics
            total_pairs = sum(len(pairs) for pairs in minimal_pairs.values())
            logger.info(f"Converted to {total_pairs} minimal pairs across {len(minimal_pairs)} phenomena")
            for phenomenon, pairs in sorted(minimal_pairs.items()):
                logger.debug(f"  {phenomenon}: {len(pairs)} pairs")

            return dict(minimal_pairs)
        except Exception as e:
            logger.warning(f"Could not load external MultiBLiMP dataset: {e}")
            logger.info("Will fall back to built-in minimal pairs")
            return None
