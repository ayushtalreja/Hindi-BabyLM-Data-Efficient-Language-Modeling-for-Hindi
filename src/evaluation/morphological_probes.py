"""
Morphological Probes for Hindi Language Models

This module implements comprehensive morphological probing tasks to analyze
what morphological information is encoded in language model representations.
Uses linear probing methodology to test for various morphological features.

Probe Types:
- Case: ergative, nominative, accusative, dative, ablative, locative, instrumental, genitive
- Number: singular, plural
- Gender: masculine, feminine
- Tense: present, past, future
- Person: 1st, 2nd, 3rd
- Aspect: perfective, imperfective, habitual
- Mood: indicative, imperative, subjunctive
- Voice: active, passive, causative
- Honorific: non-honorific, honorific, high-honorific
- Definiteness: definite, indefinite

Reference: Conneau et al. (2018) "What you can cram into a single $&!#* vector"
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MorphologicalProbe:
    """
    Comprehensive morphological probing suite for Hindi

    Features:
    - 10+ morphological probe tasks
    - Layer-wise probing analysis
    - Linear classifier methodology
    - Comprehensive test data
    - Statistical analysis
    - Per-class metrics
    """

    def __init__(self, model, tokenizer, config: Optional[Dict] = None):
        """
        Initialize morphological probe suite

        Args:
            model: Language model to probe
            tokenizer: Tokenizer for the model
            config: Optional configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or {}

        # Device setup
        self.device = next(model.parameters()).device
        logger.info(f"Morphological probe initialized on device: {self.device}")

        # Probe tasks
        self.probe_tasks = [
            'case_detection',
            'number_detection',
            'gender_detection',
            'tense_detection',
            'person_detection',
            'aspect_detection',
            'mood_detection',
            'voice_detection',
            'honorific_detection',
            'definiteness_detection'
        ]

        # Configuration
        self.probe_layers = self.config.get('probe_layers', 'all')  # 'all' or list of layer indices
        self.test_size = self.config.get('test_size', 0.2)
        self.random_state = self.config.get('random_state', 42)

        logger.info(f"Initialized probes for {len(self.probe_tasks)} morphological tasks")

    def run_all_probes(self, layer_wise: bool = True) -> Dict[str, Dict]:
        """
        Run all morphological probe tasks

        Args:
            layer_wise: Whether to probe each layer separately

        Returns:
            Dictionary mapping probe names to results
        """
        logger.info("Starting morphological probing suite...")
        results = {}

        for task in self.probe_tasks:
            logger.info(f"\nRunning {task}...")

            try:
                if layer_wise:
                    task_results = self._run_probe_layerwise(task)
                else:
                    task_results = self._run_probe_single_layer(task, layer=-1)

                results[task] = task_results

                # Log summary
                if layer_wise:
                    best_layer = max(task_results['layer_results'].items(),
                                   key=lambda x: x[1]['accuracy'])
                    logger.info(f"{task} - Best layer: {best_layer[0]}, Accuracy: {best_layer[1]['accuracy']:.4f}")
                else:
                    logger.info(f"{task} - Accuracy: {task_results['accuracy']:.4f}")

            except Exception as e:
                logger.error(f"Error running {task}: {str(e)}")
                results[task] = {'error': str(e), 'status': 'failed'}

        # Compute overall statistics
        results['overall'] = self._compute_overall_metrics(results)

        logger.info("\n" + "="*60)
        logger.info("Morphological Probing Complete")
        logger.info(f"Average Accuracy: {results['overall']['average_accuracy']:.4f}")
        logger.info("="*60)

        return results

    def _run_probe_layerwise(self, task: str) -> Dict:
        """
        Run probe on each layer of the model

        Args:
            task: Name of the probe task

        Returns:
            Dictionary with layer-wise results
        """
        # Get probe data
        sentences, target_positions, labels = self._get_probe_data(task)

        if len(sentences) == 0:
            logger.warning(f"No data for {task}")
            return {'status': 'no_data'}

        # Get number of layers
        num_layers = self._get_num_layers()

        layer_results = {}

        for layer_idx in tqdm(range(num_layers), desc=f"Probing layers for {task}"):
            # Extract representations from this layer
            representations = self._extract_representations(
                sentences, target_positions, layer=layer_idx
            )

            # Train and evaluate probe
            probe_results = self._train_and_evaluate_probe(
                representations, labels, task
            )

            layer_results[layer_idx] = probe_results

        # Find best layer
        best_layer = max(layer_results.items(), key=lambda x: x[1]['accuracy'])

        results = {
            'task': task,
            'layer_results': layer_results,
            'best_layer': best_layer[0],
            'best_accuracy': best_layer[1]['accuracy'],
            'num_layers': num_layers
        }

        return results

    def _run_probe_single_layer(self, task: str, layer: int = -1) -> Dict:
        """
        Run probe on a single layer

        Args:
            task: Name of the probe task
            layer: Layer index (-1 for last layer)

        Returns:
            Dictionary with results
        """
        # Get probe data
        sentences, target_positions, labels = self._get_probe_data(task)

        if len(sentences) == 0:
            logger.warning(f"No data for {task}")
            return {'status': 'no_data'}

        # Extract representations
        representations = self._extract_representations(
            sentences, target_positions, layer=layer
        )

        # Train and evaluate probe
        results = self._train_and_evaluate_probe(representations, labels, task)
        results['task'] = task
        results['layer'] = layer

        return results

    def _extract_representations(self, sentences: List[str],
                                target_positions: List[int],
                                layer: int = -1) -> torch.Tensor:
        """
        Extract contextualized representations for target words

        Args:
            sentences: List of sentences
            target_positions: Target word positions
            layer: Layer to extract from (-1 for last)

        Returns:
            Tensor of representations [num_examples, hidden_size]
        """
        representations = []

        self.model.eval()

        with torch.no_grad():
            for sentence, pos in zip(sentences, target_positions):
                # Tokenize
                try:
                    inputs = self.tokenizer(
                        sentence,
                        return_tensors='pt',
                        padding=False,
                        truncation=True,
                        max_length=128
                    )
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}

                    # Get hidden states
                    outputs = self.model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states  # Tuple of (num_layers+1, batch, seq_len, hidden_size)

                    # Extract from specified layer
                    layer_idx = layer if layer >= 0 else len(hidden_states) + layer
                    target_repr = hidden_states[layer_idx][0, pos, :]  # [hidden_size]

                    representations.append(target_repr.cpu())

                except Exception as e:
                    logger.warning(f"Error extracting representation: {e}")
                    # Use zero vector as fallback
                    hidden_size = self.model.config.hidden_size if hasattr(self.model, 'config') else 768
                    representations.append(torch.zeros(hidden_size))

        return torch.stack(representations)

    def _train_and_evaluate_probe(self, representations: torch.Tensor,
                                  labels: List[str], task: str) -> Dict:
        """
        Train and evaluate linear probe

        Args:
            representations: Tensor of representations
            labels: List of labels
            task: Task name

        Returns:
            Dictionary with evaluation metrics
        """
        # Convert to numpy
        X = representations.cpu().numpy()
        y = np.array(labels)

        # Determine appropriate test size based on dataset and number of classes
        num_samples = len(y)
        num_classes = len(np.unique(y))

        # Calculate minimum test samples needed for stratified split
        min_test_samples = num_classes  # At least 1 sample per class

        # Determine test_size (use proportion if dataset is large enough, otherwise use absolute number)
        if isinstance(self.test_size, float):
            # test_size is a proportion (e.g., 0.2)
            proposed_test_size = int(self.test_size * num_samples)

            if proposed_test_size < min_test_samples:
                # Dataset too small for this proportion, use absolute number
                test_size_to_use = max(min_test_samples, min(num_samples // 3, 10))
                logger.warning(
                    f"{task}: Dataset size ({num_samples}) too small for test_size={self.test_size}. "
                    f"Using {test_size_to_use} test samples instead."
                )
            else:
                test_size_to_use = self.test_size
        else:
            # test_size is already an absolute number
            test_size_to_use = self.test_size

        # Check if stratification is possible
        # For stratified split, each class needs at least 2 samples (1 train, 1 test)
        class_counts = pd.Series(y).value_counts()
        can_stratify = all(count >= 2 for count in class_counts) and num_samples >= num_classes * 2

        if not can_stratify:
            logger.warning(
                f"{task}: Cannot stratify - some classes have < 2 samples. "
                f"Using random split instead."
            )
            stratify_param = None
        else:
            stratify_param = y if num_classes > 1 else None

        # Split train/test
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_to_use, random_state=self.random_state,
                stratify=stratify_param
            )
        except ValueError as e:
            # If stratification still fails, fall back to non-stratified split
            logger.warning(f"{task}: Stratified split failed ({e}), using random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_to_use, random_state=self.random_state,
                stratify=None
            )

        # Train logistic regression probe
        probe = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
            multi_class='multinomial'
        )

        try:
            probe.fit(X_train, y_train)

            # Evaluate
            y_pred = probe.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            # Per-class metrics
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            results = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'num_train': len(X_train),
                'num_test': len(X_test),
                'num_classes': len(np.unique(y)),
                'classification_report': report
            }

        except Exception as e:
            logger.error(f"Error training probe: {e}")
            results = {
                'accuracy': 0.0,
                'error': str(e)
            }

        return results

    def _get_probe_data(self, task: str) -> Tuple[List[str], List[int], List[str]]:
        """
        Get probe data for a specific task

        Args:
            task: Task name

        Returns:
            Tuple of (sentences, target_positions, labels)
        """
        task_methods = {
            'case_detection': self._create_case_probe_data,
            'number_detection': self._create_number_probe_data,
            'gender_detection': self._create_gender_probe_data,
            'tense_detection': self._create_tense_probe_data,
            'person_detection': self._create_person_probe_data,
            'aspect_detection': self._create_aspect_probe_data,
            'mood_detection': self._create_mood_probe_data,
            'voice_detection': self._create_voice_probe_data,
            'honorific_detection': self._create_honorific_probe_data,
            'definiteness_detection': self._create_definiteness_probe_data
        }

        if task in task_methods:
            return task_methods[task]()
        else:
            logger.warning(f"Unknown task: {task}")
            return [], [], []

    def _get_num_layers(self) -> int:
        """Get number of layers in the model"""
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'num_hidden_layers'):
            return self.model.config.num_hidden_layers + 1  # +1 for embedding layer
        else:
            return 13  # Default for BERT-base-like models

    def _create_case_probe_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Create probe data for case detection"""
        data = [
            # Ergative (ने) - 10 examples
            ("राम ने किताब पढ़ी", 0, "ergative"),
            ("सीता ने खाना बनाया", 0, "ergative"),
            ("लड़के ने गाना गाया", 0, "ergative"),
            ("उसने पत्र लिखा", 0, "ergative"),
            ("मैंने फिल्म देखी", 0, "ergative"),
            ("बच्चों ने खेल खेला", 0, "ergative"),
            ("शिक्षक ने पाठ पढ़ाया", 0, "ergative"),
            ("माँ ने रोटी बनाई", 0, "ergative"),
            ("पिता ने काम किया", 0, "ergative"),
            ("लड़की ने गीत गाया", 0, "ergative"),

            # Nominative (no marker) - 10 examples
            ("राम जाता है", 0, "nominative"),
            ("लड़का खेलता है", 0, "nominative"),
            ("बच्चा सोता है", 0, "nominative"),
            ("कुत्ता भौंकता है", 0, "nominative"),
            ("सीता हंसती है", 0, "nominative"),
            ("लड़की पढ़ती है", 0, "nominative"),
            ("मोहन आता है", 0, "nominative"),
            ("शेर दहाड़ता है", 0, "nominative"),
            ("पक्षी उड़ता है", 0, "nominative"),
            ("बिल्ली म्याऊं करती है", 0, "nominative"),

            # Accusative (को) - 10 examples
            ("मैंने राम को देखा", 2, "accusative"),
            ("उसने बच्चे को बुलाया", 2, "accusative"),
            ("शिक्षक ने छात्र को पढ़ाया", 2, "accusative"),
            ("मैंने सीता को पहचाना", 2, "accusative"),
            ("पिता ने बेटे को समझाया", 2, "accusative"),
            ("लड़के ने मित्र को गाना सुनाया", 2, "accusative"),
            ("मैंने तुम्हें देखा", 2, "accusative"),
            ("उसने मुझे बताया", 2, "accusative"),
            ("राम ने लक्ष्मण को बुलाया", 2, "accusative"),
            ("सीता ने राधा को देखा", 2, "accusative"),

            # Dative (को) - 10 examples
            ("मैंने राम को किताब दी", 2, "dative"),
            ("उसने मुझे पत्र भेजा", 2, "dative"),
            ("शिक्षक ने छात्र को पुरस्कार दिया", 2, "dative"),
            ("मैंने बच्चे को खिलौना दिया", 2, "dative"),
            ("सीता ने गीता को फूल दिए", 2, "dative"),
            ("पिता ने बेटे को उपहार दिया", 2, "dative"),
            ("मैंने तुम्हें पैसे दिए", 2, "dative"),
            ("राम ने लक्ष्मण को सलाह दी", 2, "dative"),
            ("माँ ने बच्चे को दूध दिया", 2, "dative"),
            ("मित्र ने मुझे किताब दी", 2, "dative"),

            # Ablative (से - source/origin) - 10 examples
            ("राम से बात हुई", 0, "ablative"),
            ("मैं घर से आया", 1, "ablative"),
            ("दिल्ली से मुंबई", 0, "ablative"),
            ("वह स्कूल से आई", 1, "ablative"),
            ("बच्चे पार्क से लौटे", 1, "ablative"),
            ("मैं बाजार से आ रहा हूं", 1, "ablative"),
            ("वह दफ्तर से जा रहा है", 1, "ablative"),
            ("लड़की मंदिर से आई", 1, "ablative"),
            ("हम शहर से गांव गए", 1, "ablative"),
            ("वह विदेश से लौटा", 1, "ablative"),

            # Locative (में/पर) - 10 examples
            ("घर में बच्चे हैं", 0, "locative"),
            ("मेज पर किताब है", 0, "locative"),
            ("कमरे में बिस्तर है", 0, "locative"),
            ("बगीचे में फूल हैं", 0, "locative"),
            ("पेड़ पर पक्षी बैठा है", 0, "locative"),
            ("दीवार पर तस्वीर है", 0, "locative"),
            ("कुर्सी पर लड़का बैठा है", 0, "locative"),
            ("बक्से में किताबें हैं", 0, "locative"),
            ("भारत में लोग रहते हैं", 0, "locative"),
            ("दिल्ली में ताजमहल नहीं है", 0, "locative"),

            # Instrumental (से - instrument) - 10 examples
            ("चाकू से काटो", 0, "instrumental"),
            ("कलम से लिखो", 0, "instrumental"),
            ("हाथ से खाओ", 0, "instrumental"),
            ("चम्मच से खाओ", 0, "instrumental"),
            ("ब्रश से साफ करो", 0, "instrumental"),
            ("तलवार से लड़ो", 0, "instrumental"),
            ("कैंची से काटो", 0, "instrumental"),
            ("हथौड़ी से ठोको", 0, "instrumental"),
            ("पेंसिल से लिखो", 0, "instrumental"),
            ("रस्सी से बांधो", 0, "instrumental"),

            # Genitive (का/की/के) - 10 examples
            ("राम की किताब", 0, "genitive"),
            ("लड़के का नाम", 0, "genitive"),
            ("बच्चों के खिलौने", 0, "genitive"),
            ("सीता की साड़ी", 0, "genitive"),
            ("शिक्षक का घर", 0, "genitive"),
            ("लड़की के बाल", 0, "genitive"),
            ("पिता की कार", 0, "genitive"),
            ("भारत की राजधानी", 0, "genitive"),
            ("मोहन के मित्र", 0, "genitive"),
            ("माँ का प्यार", 0, "genitive"),
        ]

        sentences, positions, labels = zip(*data)
        return list(sentences), list(positions), list(labels)

    def _create_number_probe_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Create probe data for number detection"""
        data = [
            # Singular
            ("लड़का खाता है", 0, "singular"),
            ("लड़की पढ़ती है", 0, "singular"),
            ("किताब मेज पर है", 0, "singular"),
            ("कुत्ता भौंकता है", 0, "singular"),
            ("बच्चा खेलता है", 0, "singular"),

            # Plural
            ("लड़के खाते हैं", 0, "plural"),
            ("लड़कियां पढ़ती हैं", 0, "plural"),
            ("किताबें मेज पर हैं", 0, "plural"),
            ("कुत्ते भौंकते हैं", 0, "plural"),
            ("बच्चे खेलते हैं", 0, "plural"),
            ("दो लड़के आए", 1, "plural"),
            ("तीन लड़कियां गईं", 1, "plural"),
            ("सभी बच्चे खेल रहे हैं", 1, "plural"),
        ]

        sentences, positions, labels = zip(*data)
        return list(sentences), list(positions), list(labels)

    def _create_gender_probe_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Create probe data for gender detection"""
        data = [
            # Masculine
            ("लड़का गया", 0, "masculine"),
            ("राम आया", 0, "masculine"),
            ("कुत्ता भौंका", 0, "masculine"),
            ("बच्चा सोया", 0, "masculine"),
            ("अच्छा लड़का", 1, "masculine"),
            ("बड़ा घर", 1, "masculine"),

            # Feminine
            ("लड़की गई", 0, "feminine"),
            ("सीता आई", 0, "feminine"),
            ("बिल्ली म्याऊं की", 0, "feminine"),
            ("बच्ची सोई", 0, "feminine"),
            ("अच्छी लड़की", 1, "feminine"),
            ("बड़ी किताब", 1, "feminine"),
        ]

        sentences, positions, labels = zip(*data)
        return list(sentences), list(positions), list(labels)

    def _create_tense_probe_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Create probe data for tense detection"""
        data = [
            # Present
            ("मैं खाता हूं", 1, "present"),
            ("वह जाता है", 1, "present"),
            ("हम पढ़ते हैं", 1, "present"),
            ("राम खेलता है", 1, "present"),

            # Past
            ("मैं खाया", 1, "past"),
            ("वह गया", 1, "past"),
            ("हम पढ़े", 1, "past"),
            ("राम खेला", 1, "past"),
            ("मैंने खाया", 1, "past"),

            # Future
            ("मैं खाऊंगा", 1, "future"),
            ("वह जाएगा", 1, "future"),
            ("हम पढ़ेंगे", 1, "future"),
            ("राम खेलेगा", 1, "future"),
        ]

        sentences, positions, labels = zip(*data)
        return list(sentences), list(positions), list(labels)

    def _create_person_probe_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Create probe data for person detection"""
        data = [
            # 1st person
            ("मैं जाता हूं", 0, "first"),
            ("हम जाते हैं", 0, "first"),
            ("मैं खाता हूं", 0, "first"),
            ("हम खाते हैं", 0, "first"),

            # 2nd person
            ("तुम जाते हो", 0, "second"),
            ("आप जाते हैं", 0, "second"),
            ("तुम खाते हो", 0, "second"),
            ("आप खाते हैं", 0, "second"),

            # 3rd person
            ("वह जाता है", 0, "third"),
            ("राम जाता है", 0, "third"),
            ("वे जाते हैं", 0, "third"),
            ("लड़के जाते हैं", 0, "third"),
        ]

        sentences, positions, labels = zip(*data)
        return list(sentences), list(positions), list(labels)

    def _create_aspect_probe_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Create probe data for aspect detection"""
        data = [
            # Perfective
            ("मैंने खाया", 1, "perfective"),
            ("उसने पढ़ा", 1, "perfective"),
            ("राम गया", 1, "perfective"),

            # Imperfective/Progressive
            ("मैं खा रहा हूं", 1, "imperfective"),
            ("वह पढ़ रहा है", 1, "imperfective"),
            ("राम जा रहा है", 1, "imperfective"),

            # Habitual
            ("मैं रोज खाता हूं", 2, "habitual"),
            ("वह हमेशा पढ़ता है", 2, "habitual"),
            ("राम अक्सर जाता है", 2, "habitual"),
        ]

        sentences, positions, labels = zip(*data)
        return list(sentences), list(positions), list(labels)

    def _create_mood_probe_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Create probe data for mood detection"""
        data = [
            # Indicative
            ("मैं जाता हूं", 1, "indicative"),
            ("वह खाता है", 1, "indicative"),
            ("राम पढ़ता है", 1, "indicative"),

            # Imperative
            ("जाओ", 0, "imperative"),
            ("खाओ", 0, "imperative"),
            ("पढ़ो", 0, "imperative"),
            ("बैठो", 0, "imperative"),

            # Subjunctive
            ("शायद वह जाए", 2, "subjunctive"),
            ("अगर मैं जाऊं", 2, "subjunctive"),
            ("वह जाए तो अच्छा", 1, "subjunctive"),
        ]

        sentences, positions, labels = zip(*data)
        return list(sentences), list(positions), list(labels)

    def _create_voice_probe_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Create probe data for voice detection"""
        data = [
            # Active - 6 examples
            ("राम ने किताब पढ़ी", 3, "active"),
            ("सीता ने खाना बनाया", 3, "active"),
            ("मैंने पत्र लिखा", 2, "active"),
            ("लड़के ने गाना गाया", 3, "active"),
            ("शिक्षक ने पाठ पढ़ाया", 3, "active"),
            ("पिता ने काम किया", 3, "active"),

            # Passive (जाना auxiliary) - 6 examples
            ("किताब पढ़ी गई", 1, "passive"),
            ("खाना बनाया गया", 1, "passive"),
            ("पत्र लिखा गया", 1, "passive"),
            ("गाना गाया गया", 1, "passive"),
            ("काम किया गया", 1, "passive"),
            ("पाठ पढ़ाया गया", 1, "passive"),

            # Causative (वाना/लाना) - 6 examples
            ("मैंने राम से किताब पढ़वाई", 4, "causative"),
            ("उसने मुझसे खाना बनवाया", 3, "causative"),
            ("शिक्षक ने छात्र से पाठ पढ़वाया", 4, "causative"),
            ("मैंने बच्चे से काम करवाया", 4, "causative"),
            ("उसने मुझसे पत्र लिखवाया", 4, "causative"),
            ("राम ने सीता से गाना गवाया", 4, "causative"),
        ]

        sentences, positions, labels = zip(*data)
        return list(sentences), list(positions), list(labels)

    def _create_honorific_probe_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Create probe data for honorific detection"""
        data = [
            # Non-honorific (तू) - 6 examples
            ("तू जाता है", 0, "non_honorific"),
            ("तू खाता है", 0, "non_honorific"),
            ("तू पढ़ता है", 0, "non_honorific"),
            ("तू खेलता है", 0, "non_honorific"),
            ("तू सोता है", 0, "non_honorific"),
            ("तू दौड़ता है", 0, "non_honorific"),

            # Mid-honorific (तुम) - 6 examples
            ("तुम जाते हो", 0, "honorific"),
            ("तुम खाते हो", 0, "honorific"),
            ("तुम पढ़ते हो", 0, "honorific"),
            ("तुम खेलते हो", 0, "honorific"),
            ("तुम सोते हो", 0, "honorific"),
            ("तुम दौड़ते हो", 0, "honorific"),

            # High-honorific (आप) - 6 examples
            ("आप जाते हैं", 0, "high_honorific"),
            ("आप खाते हैं", 0, "high_honorific"),
            ("गुरुजी पढ़ाते हैं", 0, "high_honorific"),
            ("आप पढ़ते हैं", 0, "high_honorific"),
            ("आप खेलते हैं", 0, "high_honorific"),
            ("महाराज बैठे हैं", 0, "high_honorific"),
        ]

        sentences, positions, labels = zip(*data)
        return list(sentences), list(positions), list(labels)

    def _create_definiteness_probe_data(self) -> Tuple[List[str], List[int], List[str]]:
        """Create probe data for definiteness detection"""
        data = [
            # Definite (को marker, specific reference)
            ("मैंने उस लड़के को देखा", 2, "definite"),
            ("वह किताब अच्छी है", 1, "definite"),
            ("यह घर बड़ा है", 1, "definite"),

            # Indefinite (no marker, non-specific)
            ("मैंने एक लड़का देखा", 2, "indefinite"),
            ("कोई किताब दे दो", 1, "indefinite"),
            ("कुछ बच्चे आए", 1, "indefinite"),
        ]

        sentences, positions, labels = zip(*data)
        return list(sentences), list(positions), list(labels)

    def _compute_overall_metrics(self, results: Dict[str, Dict]) -> Dict:
        """
        Compute overall statistics across all probes

        Args:
            results: Dictionary of per-probe results

        Returns:
            Dictionary with overall metrics
        """
        accuracies = []
        f1_scores = []

        for task, task_results in results.items():
            if task == 'overall':
                continue

            if 'best_accuracy' in task_results:
                # Layer-wise results
                accuracies.append(task_results['best_accuracy'])
                # Get f1 from best layer
                best_layer = task_results['best_layer']
                if 'f1_macro' in task_results['layer_results'][best_layer]:
                    f1_scores.append(task_results['layer_results'][best_layer]['f1_macro'])
            elif 'accuracy' in task_results:
                # Single layer results
                accuracies.append(task_results['accuracy'])
                if 'f1_macro' in task_results:
                    f1_scores.append(task_results['f1_macro'])

        overall = {
            'average_accuracy': np.mean(accuracies) if accuracies else 0.0,
            'std_accuracy': np.std(accuracies) if accuracies else 0.0,
            'min_accuracy': np.min(accuracies) if accuracies else 0.0,
            'max_accuracy': np.max(accuracies) if accuracies else 0.0,
            'average_f1_macro': np.mean(f1_scores) if f1_scores else 0.0,
            'tasks_evaluated': len(accuracies),
            'accuracies_by_task': {
                task: (results[task].get('best_accuracy') or results[task].get('accuracy', 0))
                for task in self.probe_tasks
                if task in results
            }
        }

        return overall


# For backward compatibility
def run_morphological_probes(model, tokenizer, config: Optional[Dict] = None,
                            layer_wise: bool = True) -> Dict:
    """
    Convenience function to run morphological probes

    Args:
        model: Model to probe
        tokenizer: Tokenizer
        config: Optional configuration
        layer_wise: Whether to probe each layer

    Returns:
        Probe results
    """
    prober = MorphologicalProbe(model, tokenizer, config)
    return prober.run_all_probes(layer_wise=layer_wise)
