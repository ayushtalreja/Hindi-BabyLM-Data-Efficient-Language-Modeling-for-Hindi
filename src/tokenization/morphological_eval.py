import pandas as pd
from typing import List, Dict, Tuple

class MorphologicalEvaluator:
    def __init__(self):
        # Load Hindi morphological patterns
        self.inflection_patterns = self.load_inflection_patterns()
        self.compound_patterns = self.load_compound_patterns()
    
    def load_inflection_patterns(self) -> Dict:
        """Load common Hindi inflection patterns for evaluation"""
        # Common patterns: -ों (plural), -ने (ergative), -को (dative), etc.
        return {
            # Nominal morphology
            "plural": ["-ों", "-ें", "-यां", "-इयां"],
            "ergative": ["-ने"],
            "dative": ["-को"],
            "locative": ["-में", "-पर"],
            "ablative": ["-से"],
            "genitive": ["-का", "-के", "-की"],
            "instrumental": ["-से"],

            # Verbal morphology
            "habitual": ["-ता", "-ती", "-ते"],
            "perfective": ["-ा", "-ी", "-े"],
            "future": ["-ेगा", "-ेगी", "-ेगे", "-ूंगा", "-ूंगी"],
            "progressive": ["-रहा", "-रही", "-रहे"],
            "imperative": ["-ो", "-ना", "-िए"],

            # Derivational morphology
            "agentive": ["-वाला", "-वाली", "-वाले"],
            "abstract_noun": ["-पन", "-त्व", "-ई"],
            "causative": ["-वा", "-ला"],

            # Gender/number agreement
            "masculine_sg": ["-ा"],
            "feminine_sg": ["-ी"],
            "plural_oblique": ["-ों", "-ें"]
        }

    def load_compound_patterns(self) -> Dict:
        """Load compound word patterns for Hindi"""
        return {
            "noun_noun": [
                "रेलगाड़ी",      # train (rail-vehicle)
                "पाठशाला",      # school (lesson-hall)
                "जन्मदिन",      # birthday (birth-day)
                "विद्यालय",     # school (knowledge-place)
                "राजमार्ग",     # highway (king-road)
                "लोकतंत्र",     # democracy (people-system)
            ],
            "adj_noun": [
                "महापुरुष",     # great man
                "नवयुवक",       # new youth
                "पूर्णचंद्र",     # full moon
                "महाराज",       # great king
            ],
            "verb_noun": [
                "पढ़ाई",        # studying (read-nominalization)
                "लिखाई",        # writing (write-nominalization)
                "सिलाई",        # sewing (sew-nominalization)
            ],
            "reduplication": [
                "धीरे-धीरे",     # slowly-slowly (gradual)
                "बार-बार",      # time-time (repeatedly)
                "कभी-कभी",      # sometime-sometime (occasionally)
            ],
        }

    def evaluate_morphological_preservation(self, tokenizer, test_words: List[str]) -> Dict:
        """Evaluate how well tokenizer preserves morphological structure"""
        results = {
            "over_segmentation": 0,  # Morphemes split incorrectly
            "under_segmentation": 0,  # Morphemes not split when they should be
            "correct_segmentation": 0
        }
        
        for word in test_words:
            tokens = tokenizer.tokenize(word)
            # Analyze morphological correctness
            score = self.score_morphological_tokenization(word, tokens)
            results[score] += 1
        
        return results
    
    def score_morphological_tokenization(self, word: str, tokens: List[str]) -> str:
        """Score individual word tokenization quality"""
        # Check if word has known morphological structure
        has_suffix = False
        expected_splits = 1  # Base form

        # Check for known suffixes
        for category, suffixes in self.inflection_patterns.items():
            for suffix in suffixes:
                if word.endswith(suffix):
                    has_suffix = True
                    expected_splits = 2  # Root + suffix
                    break
            if has_suffix:
                break

        num_tokens = len(tokens)

        # Evaluate tokenization
        if has_suffix:
            # Word should be split into root + suffix
            if num_tokens == 2:
                return "correct_segmentation"
            elif num_tokens > 2:
                return "over_segmentation"
            else:
                return "under_segmentation"
        else:
            # Simple word should not be split
            if num_tokens == 1:
                return "correct_segmentation"
            elif num_tokens > 1:
                return "over_segmentation"
            else:
                return "under_segmentation"
    
    def create_morphological_test_set(self) -> List[str]:
        """Create test set with known morphological patterns"""
        test_words = []

        # Expanded base nouns (mix of masculine and feminine)
        base_nouns = [
            "लड़का",    # boy (m)
            "लड़की",    # girl (f)
            "किताब",    # book (f)
            "घर",       # house (m)
            "पानी",     # water (m)
            "स्कूल",    # school (m)
            "दोस्त",    # friend (m)
            "माता",     # mother (f)
            "पिता",     # father (m)
            "शिक्षक",   # teacher (m)
            "शिक्षिका", # teacher (f)
            "बच्चा",    # child (m)
            "कमरा",     # room (m)
            "मेज",      # table (f)
            "कुर्सी",    # chair (f)
            "दरवाजा",   # door (m)
            "खिड़की",    # window (f)
            "गाड़ी",     # vehicle (f)
            "सड़क",     # road (f)
            "शहर",      # city (m)
        ]

        # Verb roots for verbal morphology tests
        verb_roots = [
            "पढ़",      # read
            "लिख",      # write
            "खा",       # eat
            "जा",       # go
            "आ",        # come
            "देख",      # see
            "सुन",      # hear
            "बोल",      # speak
            "कर",       # do
            "दे",       # give
        ]

        # Add base words (unmorphed)
        test_words.extend(base_nouns)

        # Add nominal inflections
        for base in base_nouns[:10]:  # Use first 10 to keep test set manageable
            # Plural forms
            test_words.append(base + "ों")
            test_words.append(base + "ें")

            # Case markers
            test_words.append(base + "ने")   # ergative
            test_words.append(base + "को")   # dative
            test_words.append(base + "में")   # locative
            test_words.append(base + "पर")   # locative
            test_words.append(base + "से")   # ablative/instrumental
            test_words.append(base + "का")   # genitive masculine
            test_words.append(base + "के")   # genitive masculine plural
            test_words.append(base + "की")   # genitive feminine

        # Add verbal inflections
        for verb in verb_roots[:8]:  # Use first 8 verbs
            # Habitual aspect
            test_words.append(verb + "ता")   # masculine singular
            test_words.append(verb + "ती")   # feminine singular
            test_words.append(verb + "ते")   # masculine plural

            # Future tense
            test_words.append(verb + "ेगा")  # masculine singular
            test_words.append(verb + "ेगी")  # feminine singular

            # Progressive aspect
            test_words.append(verb + "रहा")  # masculine singular
            test_words.append(verb + "रही")  # feminine singular
            test_words.append(verb + "रहे")  # masculine plural

        # Add derivational morphology examples
        derivational_bases = ["दूध", "बच्चा", "अच्छा", "मीठा", "पागल"]
        for base in derivational_bases:
            test_words.append(base + "वाला")  # agentive masculine
            test_words.append(base + "वाली")  # agentive feminine

        # Abstract nouns
        test_words.extend([
            "मीठापन",    # sweetness (मीठा + पन)
            "देवत्व",     # divinity (देव + त्व)
            "सुंदरता",    # beauty (सुंदर + ता)
            "बचपन",      # childhood (बच्चा + पन)
        ])

        # Add all compound word examples
        for compounds in self.compound_patterns.values():
            test_words.extend(compounds)

        return test_words