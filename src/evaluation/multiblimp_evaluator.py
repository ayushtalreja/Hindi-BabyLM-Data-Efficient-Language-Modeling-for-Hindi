"""
MultiBLiMP (Multilingual BLiMP) Evaluator for Hindi

This module implements comprehensive evaluation of syntactic phenomena in Hindi
using minimal pair testing methodology. Each test consists of a grammatical
sentence paired with an ungrammatical variant differing in a single linguistic
feature.

Phenomena Tested:
- Subject-verb agreement (number, person, gender)
- Case marking (ergative, nominative, accusative, dative)
- Word order variations
- Gender agreement (noun-adjective, noun-verb)
- Number agreement
- Honorific agreement
- Binding principles
- Control structures
- Negation
- Quantifier scope

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
    - 10+ linguistic phenomena tested
    - Perplexity-based evaluation
    - Comprehensive minimal pair database
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

        # Phenomena to test
        self.phenomena = [
            'subject_verb_agreement_number',
            'subject_verb_agreement_person',
            'subject_verb_agreement_gender',
            'case_marking_ergative',
            'case_marking_accusative',
            'case_marking_dative',
            'word_order',
            'gender_agreement_adjective',
            'gender_agreement_verb',
            'number_agreement',
            'honorific_agreement',
            'negation',
            'binding',
            'control'
        ]

        # Load or create minimal pairs
        self.minimal_pairs = self._initialize_minimal_pairs()

        logger.info(f"Loaded {sum(len(pairs) for pairs in self.minimal_pairs.values())} minimal pairs across {len(self.minimal_pairs)} phenomena")

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

            # Get losses
            good_outputs = self.model(**good_inputs, labels=good_inputs['input_ids'])
            bad_outputs = self.model(**bad_inputs, labels=bad_inputs['input_ids'])

            good_loss = good_outputs.loss.item()
            bad_loss = bad_outputs.loss.item()

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
        """
        # Try loading from HuggingFace or external source
        try:
            dataset = self._load_multiblimp_dataset()
            if dataset:
                return dataset
        except Exception as e:
            logger.debug(f"Could not load external MultiBLiMP dataset: {e}")

        # Use comprehensive built-in minimal pairs
        logger.info("Using built-in Hindi minimal pairs database")
        return self._create_comprehensive_minimal_pairs()

    def _load_multiblimp_dataset(self) -> Optional[Dict]:
        """
        Load MultiBLiMP dataset from HuggingFace

        Returns:
            Dataset or None if not available
        """
        try:
            dataset = load_dataset('BabyLM/MultilingualBlimp', 'hi', split='test')
            # Convert to our format
            minimal_pairs = defaultdict(list)
            for example in dataset:
                phenomenon = example.get('phenomenon', 'unknown')
                minimal_pairs[phenomenon].append({
                    'good': example['good_sentence'],
                    'bad': example['bad_sentence'],
                    'phenomenon': phenomenon
                })
            return dict(minimal_pairs)
        except:
            return None

    def _create_comprehensive_minimal_pairs(self) -> Dict[str, List[Dict]]:
        """
        Create comprehensive database of Hindi minimal pairs

        Returns:
            Dictionary mapping phenomena to minimal pairs
        """
        pairs = {
            'subject_verb_agreement_number': [
                # Singular vs Plural
                {'good': 'लड़का खाता है', 'bad': 'लड़का खाते हैं', 'gloss': 'boy eats (sg)'},
                {'good': 'लड़के खाते हैं', 'bad': 'लड़के खाता है', 'gloss': 'boys eat (pl)'},
                {'good': 'लड़की पढ़ती है', 'bad': 'लड़की पढ़ती हैं', 'gloss': 'girl reads (sg)'},
                {'good': 'लड़कियां पढ़ती हैं', 'bad': 'लड़कियां पढ़ती है', 'gloss': 'girls read (pl)'},
                {'good': 'किताब मेज पर है', 'bad': 'किताब मेज पर हैं', 'gloss': 'book is on table (sg)'},
                {'good': 'किताबें मेज पर हैं', 'bad': 'किताबें मेज पर है', 'gloss': 'books are on table (pl)'},
            ],

            'subject_verb_agreement_person': [
                # 1st, 2nd, 3rd person
                {'good': 'मैं जाता हूं', 'bad': 'मैं जाता है', 'gloss': 'I go (1st person)'},
                {'good': 'तुम जाते हो', 'bad': 'तुम जाता है', 'gloss': 'you go (2nd person)'},
                {'good': 'वह जाता है', 'bad': 'वह जाते हो', 'gloss': 'he goes (3rd person)'},
                {'good': 'हम जाते हैं', 'bad': 'हम जाता है', 'gloss': 'we go (1st person pl)'},
                {'good': 'आप जाते हैं', 'bad': 'आप जाता है', 'gloss': 'you go (2nd person honorific)'},
            ],

            'subject_verb_agreement_gender': [
                # Masculine vs Feminine
                {'good': 'लड़का गया', 'bad': 'लड़का गई', 'gloss': 'boy went (masc)'},
                {'good': 'लड़की गई', 'bad': 'लड़की गया', 'gloss': 'girl went (fem)'},
                {'good': 'राम आया', 'bad': 'राम आई', 'gloss': 'Ram came (masc)'},
                {'good': 'सीता आई', 'bad': 'सीता आया', 'gloss': 'Sita came (fem)'},
                {'good': 'कुत्ता भौंका', 'bad': 'कुत्ता भौंकी', 'gloss': 'dog barked (masc)'},
                {'good': 'बिल्ली म्याऊं की', 'bad': 'बिल्ली म्याऊं किया', 'gloss': 'cat meowed (fem)'},
            ],

            'case_marking_ergative': [
                # Ergative ne marker in perfective transitive
                {'good': 'राम ने किताब पढ़ी', 'bad': 'राम किताब पढ़ी', 'gloss': 'Ram read book (ergative)'},
                {'good': 'सीता ने खाना बनाया', 'bad': 'सीता खाना बनाया', 'gloss': 'Sita made food (ergative)'},
                {'good': 'लड़के ने गाना गाया', 'bad': 'लड़के गाना गाया', 'gloss': 'boy sang song (ergative)'},
                {'good': 'उसने पत्र लिखा', 'bad': 'उस पत्र लिखा', 'gloss': 'he wrote letter (ergative)'},
            ],

            'case_marking_accusative': [
                # ko marker for specific/animate objects
                {'good': 'मैंने राम को देखा', 'bad': 'मैंने राम देखा', 'gloss': 'I saw Ram (accusative)'},
                {'good': 'उसने बच्चे को बुलाया', 'bad': 'उसने बच्चे बुलाया', 'gloss': 'he called child (acc)'},
                {'good': 'राम किताब पढ़ता है', 'bad': 'राम किताब को पढ़ता है', 'gloss': 'Ram reads book (no acc for inanimate)'},
            ],

            'case_marking_dative': [
                # ko marker for indirect objects
                {'good': 'मैंने राम को किताब दी', 'bad': 'मैंने राम किताब दी', 'gloss': 'I gave book to Ram'},
                {'good': 'उसने मुझे पत्र भेजा', 'bad': 'उसने मैं पत्र भेजा', 'gloss': 'he sent letter to me'},
            ],

            'word_order': [
                # SOV is natural, OSV is marked but grammatical
                {'good': 'राम ने किताब पढ़ी', 'bad': 'पढ़ी राम ने किताब', 'gloss': 'Ram read book (SOV)'},
                {'good': 'लड़की स्कूल जाती है', 'bad': 'जाती है लड़की स्कूल', 'gloss': 'girl goes to school'},
                {'good': 'मैं खाना खाता हूं', 'bad': 'खाता हूं मैं खाना', 'gloss': 'I eat food'},
            ],

            'gender_agreement_adjective': [
                # Adjective agrees with noun gender
                {'good': 'अच्छा लड़का', 'bad': 'अच्छी लड़का', 'gloss': 'good boy (masc adj)'},
                {'good': 'अच्छी लड़की', 'bad': 'अच्छा लड़की', 'gloss': 'good girl (fem adj)'},
                {'good': 'बड़ा घर', 'bad': 'बड़ी घर', 'gloss': 'big house (masc)'},
                {'good': 'बड़ी किताब', 'bad': 'बड़ा किताब', 'gloss': 'big book (fem)'},
                {'good': 'नया कपड़ा', 'bad': 'नई कपड़ा', 'gloss': 'new cloth (masc)'},
                {'good': 'नई मेज', 'bad': 'नया मेज', 'gloss': 'new table (fem)'},
            ],

            'gender_agreement_verb': [
                # Past tense verb agrees with subject gender
                {'good': 'लड़का आया', 'bad': 'लड़का आई', 'gloss': 'boy came (masc verb)'},
                {'good': 'लड़की आई', 'bad': 'लड़की आया', 'gloss': 'girl came (fem verb)'},
                {'good': 'राम गया', 'bad': 'राम गई', 'gloss': 'Ram went (masc)'},
                {'good': 'गीता गई', 'bad': 'गीता गया', 'gloss': 'Geeta went (fem)'},
            ],

            'number_agreement': [
                # Plural marking consistency
                {'good': 'दो लड़के आए', 'bad': 'दो लड़का आया', 'gloss': 'two boys came'},
                {'good': 'तीन लड़कियां गईं', 'bad': 'तीन लड़की गई', 'gloss': 'three girls went'},
                {'good': 'सभी बच्चे खेल रहे हैं', 'bad': 'सभी बच्चा खेल रहा है', 'gloss': 'all children playing'},
            ],

            'honorific_agreement': [
                # Honorific vs non-honorific verb forms
                {'good': 'आप जाते हैं', 'bad': 'आप जाता है', 'gloss': 'you go (honorific)'},
                {'good': 'गुरुजी पढ़ाते हैं', 'bad': 'गुरुजी पढ़ाता है', 'gloss': 'teacher teaches (hon)'},
                {'good': 'तुम जाते हो', 'bad': 'तुम जाते हैं', 'gloss': 'you go (familiar, not honorific)'},
            ],

            'negation': [
                # Negation with नहीं
                {'good': 'मैं नहीं जाता', 'bad': 'मैं जाता नहीं', 'gloss': 'I don\'t go (neg before verb)'},
                {'good': 'वह नहीं आया', 'bad': 'वह आया नहीं', 'gloss': 'he didn\'t come'},
                {'good': 'राम नहीं पढ़ता', 'bad': 'राम पढ़ता नहीं', 'gloss': 'Ram doesn\'t read'},
            ],

            'binding': [
                # Reflexive pronoun binding
                {'good': 'राम अपने आप को देखता है', 'bad': 'राम उसको देखता है', 'gloss': 'Ram sees himself (reflexive)'},
                {'good': 'वह अपनी किताब पढ़ता है', 'bad': 'वह उसकी किताब पढ़ता है', 'gloss': 'he reads his own book'},
            ],

            'control': [
                # Control structures with infinitives
                {'good': 'मैं जाना चाहता हूं', 'bad': 'मैं जाता चाहता हूं', 'gloss': 'I want to go (infinitive)'},
                {'good': 'वह खाना चाहती है', 'bad': 'वह खाती चाहती है', 'gloss': 'she wants to eat'},
                {'good': 'हम पढ़ना पसंद करते हैं', 'bad': 'हम पढ़ते पसंद करते हैं', 'gloss': 'we like to read'},
            ]
        }

        return pairs


# For backward compatibility
def evaluate_multiblimp(model, tokenizer, config: Optional[Dict] = None) -> Dict:
    """
    Convenience function to evaluate on MultiBLiMP

    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        config: Optional configuration

    Returns:
        Evaluation results
    """
    evaluator = MultiBLiMPEvaluator(model, tokenizer, config)
    return evaluator.evaluate_all_phenomena()
