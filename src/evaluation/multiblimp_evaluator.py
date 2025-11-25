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
            model: Language model to evaluate
            tokenizer: Tokenizer for the model
            config: Optional configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}

        # Device setup
        self.device = next(model.parameters()).device
        logger.info(f"MultiBLiMP evaluator initialized on device: {self.device}")

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

        self.model.eval()

        with torch.no_grad():
            for pair in tqdm(pairs, desc=f"Evaluating {phenomenon}"):
                # Evaluate minimal pair
                is_correct, good_loss, bad_loss, loss_diff = self._evaluate_minimal_pair_detailed(
                    pair['good'],
                    pair['bad']
                )

                if is_correct:
                    correct_predictions += 1

                loss_differences.append(loss_diff)
                pair_results.append({
                    'good': pair['good'],
                    'bad': pair['bad'],
                    'correct': is_correct,
                    'good_loss': good_loss,
                    'bad_loss': bad_loss,
                    'loss_difference': loss_diff
                })

        # Compute metrics
        accuracy = correct_predictions / total_pairs if total_pairs > 0 else 0

        results = {
            'phenomenon': phenomenon,
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': total_pairs,
            'mean_loss_difference': np.mean(loss_differences) if loss_differences else 0,
            'std_loss_difference': np.std(loss_differences) if loss_differences else 0,
            'pair_results': pair_results if self.config.get('save_pair_results', False) else None
        }

        return results

    def _evaluate_minimal_pair_detailed(self, good_sentence: str, bad_sentence: str) -> Tuple[bool, float, float, float]:
        """
        Evaluate a single minimal pair with detailed metrics

        Args:
            good_sentence: Grammatical sentence
            bad_sentence: Ungrammatical sentence

        Returns:
            Tuple of (is_correct, good_loss, bad_loss, loss_difference)
        """
        try:
            # Tokenize both sentences
            good_inputs = self._tokenize_sentence(good_sentence)
            bad_inputs = self._tokenize_sentence(bad_sentence)

            # Get model outputs WITHOUT labels parameter
            good_outputs = self.model(**good_inputs)
            bad_outputs = self.model(**bad_inputs)

            # Extract logits [batch_size, seq_len, vocab_size]
            good_logits = good_outputs.logits
            bad_logits = bad_outputs.logits

            # Compute loss for good sentence
            # Shift logits and labels for next-token prediction
            good_shift_logits = good_logits[:, :-1, :].contiguous()
            good_shift_labels = good_inputs['input_ids'][:, 1:].contiguous()

            # Compute cross-entropy loss
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

        except Exception as e:
            logger.warning(f"Error evaluating pair: {e}")
            return False, float('inf'), float('inf'), 0.0

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
            }
        }

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
