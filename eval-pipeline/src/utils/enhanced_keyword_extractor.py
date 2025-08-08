#!/usr/bin/env python3
"""
Enhanced Hybrid Keyword Extractor

Advanced hybrid keyword extractor implementing language-detection-first pipeline
with weighted source content processing and multiple extraction methods.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import re
import time
from pathlib import Path
import asyncio
from collections import Counter
import json
import hashlib
import pickle
from datetime import datetime, timedelta

# Language detection
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0  # For reproducible results
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# KeyBERT imports
try:
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False

# spaCy imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# YAKE imports
try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False

# RAGAS LLM extractors
try:
    from ragas.testset.transforms.extractors.llm_based import (
        KeyphrasesExtractor, NERExtractor
    )
    RAGAS_LLM_AVAILABLE = True
except ImportError:
    RAGAS_LLM_AVAILABLE = False

# sklearn imports for cosine similarity
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Performance monitoring and caching imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelPerformanceMonitor:
    """Monitor model performance and selection effectiveness"""
    
    def __init__(self, cache_dir: str = "./cache/performance"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.performance_data = {}
        self.session_stats = {
            'model_selections': Counter(),
            'processing_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'memory_usage': []
        }
    
    def record_model_selection(self, model_name: str, selection_reason: str, 
                             processing_time: float, success: bool):
        """Record model selection and performance"""
        self.session_stats['model_selections'][model_name] += 1
        self.session_stats['processing_times'].append({
            'model': model_name,
            'time': processing_time,
            'success': success,
            'reason': selection_reason,
            'timestamp': datetime.now().isoformat()
        })
    
    def record_memory_usage(self):
        """Record current memory usage if psutil available"""
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.session_stats['memory_usage'].append({
                    'memory_mb': memory_mb,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.debug(f"Failed to record memory usage: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for current session"""
        processing_times = [p['time'] for p in self.session_stats['processing_times'] if p['success']]
        return {
            'model_selections': dict(self.session_stats['model_selections']),
            'avg_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
            'cache_hit_rate': self.session_stats['cache_hits'] / max(1, self.session_stats['cache_hits'] + self.session_stats['cache_misses']),
            'total_operations': len(self.session_stats['processing_times']),
            'success_rate': sum(1 for p in self.session_stats['processing_times'] if p['success']) / max(1, len(self.session_stats['processing_times'])),
            'peak_memory_mb': max([m['memory_mb'] for m in self.session_stats['memory_usage']], default=0)
        }


class EmbeddingCache:
    """Advanced caching system for embedding computations"""
    
    def __init__(self, cache_dir: str = "./cache/embeddings", max_size_mb: int = 512):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.memory_cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}
    
    def _get_cache_key(self, texts: List[str], model_name: str) -> str:
        """Generate cache key for texts and model"""
        content_hash = hashlib.md5(
            ("|".join(sorted(texts)) + model_name).encode('utf-8')
        ).hexdigest()
        return f"{model_name}_{content_hash}"
    
    def get_embeddings(self, texts: List[str], model_name: str) -> Optional[List]:
        """Get cached embeddings if available"""
        cache_key = self._get_cache_key(texts, model_name)
        
        # Check memory cache first
        if cache_key in self.memory_cache:
            self.cache_stats['hits'] += 1
            return self.memory_cache[cache_key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    embeddings = pickle.load(f)
                self.memory_cache[cache_key] = embeddings
                self.cache_stats['hits'] += 1
                return embeddings
            except Exception as e:
                logger.debug(f"Failed to load cached embeddings: {e}")
        
        self.cache_stats['misses'] += 1
        return None
    
    def cache_embeddings(self, texts: List[str], model_name: str, embeddings: List):
        """Cache embeddings to memory and disk"""
        cache_key = self._get_cache_key(texts, model_name)
        
        # Store in memory cache
        self.memory_cache[cache_key] = embeddings
        
        # Store in disk cache
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)
        except Exception as e:
            logger.debug(f"Failed to cache embeddings to disk: {e}")
    
    def cleanup_cache(self):
        """Clean up old cache files if size exceeds limit"""
        try:
            total_size = sum(f.stat().st_size for f in self.cache_dir.glob("*.pkl"))
            if total_size > self.max_size_mb * 1024 * 1024:
                # Remove oldest files
                files = [(f, f.stat().st_mtime) for f in self.cache_dir.glob("*.pkl")]
                files.sort(key=lambda x: x[1])  # Sort by modification time
                
                for file_path, _ in files[:len(files)//2]:  # Remove half of the files
                    file_path.unlink()
                    
                logger.info(f"🧹 Cleaned up embedding cache, removed old files")
        except Exception as e:
            logger.debug(f"Failed to cleanup cache: {e}")


class EnhancedHybridKeywordExtractor:
    """
    Advanced hybrid keyword extractor implementing language-detection-first pipeline
    with weighted source content processing and multiple extraction methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.extraction_config = config.get('testset_generation', {}).get('keyword_extraction', {})
        self.source_weights = self.extraction_config.get('source_weights', {
            'user_query': 0.40,
            'reference_contexts': 0.35, 
            'reference_answer': 0.25
        })
        
        # Initialize advanced components
        self.performance_monitor = ModelPerformanceMonitor()
        self.embedding_cache = EmbeddingCache(
            cache_dir=self.extraction_config.get('cache_dir', './cache/embeddings'),
            max_size_mb=self.extraction_config.get('max_cache_size_mb', 512)
        )
        
        # Initialize content analysis cache
        self.content_analysis_cache = {}
        
        # Domain knowledge base for enhanced validation
        self.domain_knowledge = {
            'technical_prefixes': ['auto', 'semi', 'multi', 'inter', 'intra', 'pre', 'post', 'pro', 'anti'],
            'technical_suffixes': ['tion', 'sion', 'ment', 'ness', 'able', 'ible', 'ical', 'ous', 'ive'],
            'domain_indicators': {
                'manufacturing': ['production', 'assembly', 'quality', 'inspection', 'defect'],
                'engineering': ['design', 'analysis', 'specification', 'testing', 'validation'],
                'technical': ['system', 'process', 'method', 'procedure', 'protocol']
            }
        }
        
        # Initialize extraction models
        self._init_models()
        
    def _init_models(self):
        """Initialize all extraction models based on config"""
        self.models = {}
        
        # Initialize KeyBERT models
        if KEYBERT_AVAILABLE and self.extraction_config.get('hybrid_pipeline', {}).get('extraction_methods', {}).get('keybert_enhanced', {}).get('enabled', True):
            self._init_keybert_models()
            
        # Initialize spaCy models  
        if SPACY_AVAILABLE and self.extraction_config.get('hybrid_pipeline', {}).get('extraction_methods', {}).get('spacy_nlp', {}).get('enabled', True):
            self._init_spacy_models()
            
        # Initialize YAKE
        if YAKE_AVAILABLE:
            self.yake_extractor = None  # Lazy initialization per language
            
        logger.info(f"🔧 Initialized Enhanced Hybrid Keyword Extractor with {len(self.models)} model groups")
    
    def _init_keybert_models(self):
        """Initialize KeyBERT models for different languages"""
        keybert_config = self.extraction_config.get('hybrid_pipeline', {}).get('extraction_methods', {}).get('keybert_enhanced', {})
        models_config = keybert_config.get('models', {})
        
        self.models['keybert'] = {}
        
        # Chinese model
        if 'chinese' in models_config:
            try:
                chinese_model = SentenceTransformer(models_config['chinese'])
                self.models['keybert']['chinese'] = KeyBERT(model=chinese_model)
                logger.info(f"✅ Loaded Chinese KeyBERT model: {models_config['chinese']}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Chinese KeyBERT model: {e}")
                
        # English model  
        if 'english' in models_config:
            try:
                english_model = SentenceTransformer(models_config['english'])
                self.models['keybert']['english'] = KeyBERT(model=english_model)
                logger.info(f"✅ Loaded English KeyBERT model: {models_config['english']}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load English KeyBERT model: {e}")
                
        # Multilingual fallback
        if 'multilingual' in models_config:
            try:
                multilingual_model = SentenceTransformer(models_config['multilingual'])
                self.models['keybert']['multilingual'] = KeyBERT(model=multilingual_model)
                logger.info(f"✅ Loaded Multilingual KeyBERT model: {models_config['multilingual']}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Multilingual KeyBERT model: {e}")
    
    def _init_spacy_models(self):
        """Initialize spaCy models for different languages"""
        spacy_config = self.extraction_config.get('hybrid_pipeline', {}).get('extraction_methods', {}).get('spacy_nlp', {})
        models_config = spacy_config.get('models', {})
        
        self.models['spacy'] = {}
        
        for lang, model_name in models_config.items():
            try:
                self.models['spacy'][lang] = spacy.load(model_name)
                logger.info(f"✅ Loaded spaCy {lang} model: {model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load spaCy {lang} model: {e}")
    
    def detect_language(self, text: str) -> str:
        """Detect language of text"""
        lang_config = self.extraction_config.get('hybrid_pipeline', {}).get('language_detection', {})
        
        if not lang_config.get('enabled', True):
            return lang_config.get('fallback_language', 'en')
            
        if not LANGDETECT_AVAILABLE:
            logger.warning("⚠️ langdetect not available, using fallback language")
            return lang_config.get('fallback_language', 'en')
            
        try:
            detected = detect(text)
            # Map detected languages to our supported languages
            if detected in ['zh-cn', 'zh-tw', 'zh']:
                return 'chinese'
            elif detected == 'en':
                return 'english'
            else:
                return 'multilingual'  # Use multilingual model for other languages
        except Exception as e:
            logger.warning(f"⚠️ Language detection failed: {e}, using fallback")
            return lang_config.get('fallback_language', 'en')
    
    def extract_keywords_from_sources(self, user_query: str, reference_contexts: List[str], reference_answer: str) -> Dict[str, Any]:
        """
        Extract keywords from multiple sources with weighted combination
        
        Args:
            user_query: The question/query text
            reference_contexts: List of context documents from CSV
            reference_answer: Generated answer text
            
        Returns:
            Dict containing combined keywords with metadata
        """
        logger.info("🔑 Starting hybrid keyword extraction from multiple sources")
        
        # Prepare source texts
        sources = {
            'user_query': user_query or "",
            'reference_contexts': " ".join(reference_contexts) if reference_contexts else "",
            'reference_answer': reference_answer or ""
        }
        
        # Language detection for each source
        language_detection = {}
        for source_name, text in sources.items():
            if text.strip():
                detected_lang = self.detect_language(text)
                language_detection[source_name] = {
                    'detected_language': detected_lang,
                    'text_length': len(text),
                    'character_count_chinese': len(re.findall(r'[\u4e00-\u9fff]', text)),
                    'character_count_english': len(re.findall(r'[a-zA-Z]', text))
                }
        
        # Extract keywords from each source
        source_keywords = {}
        total_weight = 0
        
        for source_name, text in sources.items():
            if not text.strip():
                continue
                
            weight = self.source_weights.get(source_name, 0.0)
            if weight <= 0:
                continue
                
            logger.info(f"🔍 Extracting keywords from {source_name} (weight: {weight})")
            keywords = self._extract_keywords_from_text(text)
            source_keywords[source_name] = {
                'keywords': keywords,
                'weight': weight,
                'text_length': len(text),
                'detected_language': language_detection.get(source_name, {}).get('detected_language', 'unknown')
            }
            total_weight += weight
        
        # Combine keywords with weighted scoring
        combined_keywords = self._combine_weighted_keywords(source_keywords, total_weight)
        
        # ADVANCED ENHANCEMENT: Analyze content characteristics
        content_analysis = self._analyze_content_characteristics(
            user_query, reference_contexts, reference_answer
        )
        
        # Post-process and filter with content context
        final_keywords, post_process_metadata = self._post_process_keywords(
            combined_keywords, content_context=content_analysis
        )
        
        # Calculate language distribution
        total_chars = sum(stats['text_length'] for stats in language_detection.values())
        total_chinese = sum(stats['character_count_chinese'] for stats in language_detection.values())
        total_english = sum(stats['character_count_english'] for stats in language_detection.values())
        
        language_distribution = {
            'chinese_percentage': round((total_chinese / total_chars * 100), 2) if total_chars > 0 else 0,
            'english_percentage': round((total_english / total_chars * 100), 2) if total_chars > 0 else 0,
            'mixed_content': total_chinese > 0 and total_english > 0
        }
        
        # Collect detailed keyword tracking
        keyword_details = []
        extraction_methods_used = set()
        
        for source_name, source_data in source_keywords.items():
            for keyword_data in source_data['keywords']:
                methods = keyword_data.get('methods', [])
                extraction_methods_used.update(methods)
                keyword_details.append({
                    'keyword': keyword_data['keyword'],
                    'source': source_name,
                    'methods': methods,
                    'score': keyword_data.get('total_score', keyword_data.get('score', 0)),
                    'language': source_data.get('detected_language', 'unknown')
                })
        
        logger.info(f"✅ Generated {len(final_keywords)} final keywords from {len(source_keywords)} sources")
        
        return {
            'keywords': final_keywords,
            'source_breakdown': source_keywords,
            'language_detection': language_detection,
            'language_distribution': language_distribution,
            'keyword_details': keyword_details,
            'content_analysis': content_analysis,  # NEW: Include content analysis
            'extraction_metadata': {
                'method': 'hybrid_weighted',
                'total_sources': len(source_keywords),
                'extraction_methods_used': list(extraction_methods_used)
            },
            'post_process_metadata': post_process_metadata
        }
    
    def _extract_keywords_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract keywords from single text using hybrid pipeline"""
        if not text.strip():
            return []
            
        # Step 1: Language detection
        language = self.detect_language(text)
        logger.debug(f"🌐 Detected language: {language}")
        
        # Step 2: Extract using multiple methods
        extraction_results = []
        
        # KeyBERT extraction
        keybert_keywords = self._extract_with_keybert(text, language)
        if keybert_keywords:
            extraction_results.extend(keybert_keywords)
            
        # spaCy NLP extraction
        spacy_keywords = self._extract_with_spacy(text, language)
        if spacy_keywords:
            extraction_results.extend(spacy_keywords)
            
        # YAKE statistical extraction
        yake_keywords = self._extract_with_yake(text, language)
        if yake_keywords:
            extraction_results.extend(yake_keywords)
            
        # Step 3: Score and rank results
        return self._score_and_rank_keywords(extraction_results, text)
    
    def _extract_with_keybert(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extract keywords using KeyBERT"""
        keybert_config = self.extraction_config.get('hybrid_pipeline', {}).get('extraction_methods', {}).get('keybert_enhanced', {})
        
        if not keybert_config.get('enabled', True) or 'keybert' not in self.models:
            return []
            
        # Select appropriate model
        model_key = language if language in self.models['keybert'] else 'multilingual'
        if model_key not in self.models['keybert']:
            model_key = next(iter(self.models['keybert']), None)
            
        if not model_key:
            return []
            
        try:
            keybert_model = self.models['keybert'][model_key]
            
            # Extract keywords with MMR for diversity
            use_mmr = keybert_config.get('use_mmr', True)
            diversity = keybert_config.get('diversity', 0.5)
            
            if use_mmr:
                keywords = keybert_model.extract_keywords(
                    text, 
                    keyphrase_ngram_range=(1, 3),
                    stop_words='english' if language == 'english' else None,
                    use_mmr=True,
                    diversity=diversity,
                    nr_candidates=20  # Use nr_candidates instead of top_k
                )
            else:
                keywords = keybert_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 3), 
                    stop_words='english' if language == 'english' else None,
                    nr_candidates=20  # Use nr_candidates instead of top_k
                )
                
            # Format results - take top 10 results
            result = []
            base_weight = keybert_config.get('weight', 0.35)
            
            for keyword, score in keywords[:10]:  # Limit to top 10
                result.append({
                    'keyword': keyword,
                    'score': score,
                    'method': 'keybert',
                    'language': language,
                    'model': model_key,
                    'weight': base_weight
                })
                
            logger.debug(f"🔸 KeyBERT extracted {len(result)} keywords")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ KeyBERT extraction failed: {e}")
            return []
    
    def _extract_with_spacy(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extract keywords using spaCy NLP"""
        spacy_config = self.extraction_config.get('hybrid_pipeline', {}).get('extraction_methods', {}).get('spacy_nlp', {})
        
        if not spacy_config.get('enabled', True) or 'spacy' not in self.models:
            return []
            
        # Select appropriate model
        model_key = language if language in self.models['spacy'] else 'english'
        if model_key not in self.models['spacy']:
            return []
            
        try:
            nlp = self.models['spacy'][model_key]
            doc = nlp(text)
            
            result = []
            base_weight = spacy_config.get('weight', 0.20)
            
            # Extract named entities
            if spacy_config.get('extract_entities', True):
                for ent in doc.ents:
                    if len(ent.text.strip()) > 2:  # Filter short entities
                        result.append({
                            'keyword': ent.text.strip(),
                            'score': 0.8,  # High confidence for NER
                            'method': 'spacy_ner',
                            'entity_type': ent.label_,
                            'language': language,
                            'weight': base_weight
                        })
            
            # Extract noun phrases
            if spacy_config.get('extract_noun_phrases', True):
                for chunk in doc.noun_chunks:
                    if len(chunk.text.strip()) > 2 and chunk.root.pos_ in ['NOUN', 'PROPN']:
                        result.append({
                            'keyword': chunk.text.strip(),
                            'score': 0.6,  # Medium confidence for noun phrases
                            'method': 'spacy_noun_phrase',
                            'pos_tag': chunk.root.pos_,
                            'language': language,
                            'weight': base_weight
                        })
            
            logger.debug(f"🔸 spaCy extracted {len(result)} keywords")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ spaCy extraction failed: {e}")
            return []
    
    def _extract_with_yake(self, text: str, language: str) -> List[Dict[str, Any]]:
        """Extract keywords using YAKE statistical method"""
        statistical_config = self.extraction_config.get('hybrid_pipeline', {}).get('extraction_methods', {}).get('statistical', {})
        
        if not statistical_config.get('enabled', True) or not YAKE_AVAILABLE:
            return []
            
        try:
            yake_config = statistical_config.get('yake_config', {})
            
            # Map language to YAKE language codes
            lang_map = {
                'chinese': 'zh',
                'english': 'en', 
                'multilingual': 'en'  # Fallback to English
            }
            yake_lang = lang_map.get(language, 'en')
            
            # Initialize YAKE extractor with proper language parameter
            kw_extractor = yake.KeywordExtractor(
                lan=yake_lang,
                n=yake_config.get('n', 3),
                dedupLim=yake_config.get('dedupLim', 0.7),
                top=20
            )
            
            keywords = kw_extractor.extract_keywords(text)
            
            result = []
            base_weight = statistical_config.get('weight', 0.20)
            
            for score, keyword in keywords[:10]:  # Top 10 keywords
                # Validate keyword is not a numeric value
                if isinstance(keyword, (int, float)):
                    logger.debug(f"🔸 YAKE: Skipping numeric keyword: {keyword}")
                    continue
                    
                if isinstance(keyword, str):
                    # Check if keyword is actually a numeric string
                    try:
                        float(keyword)
                        logger.debug(f"🔸 YAKE: Skipping numeric string keyword: {keyword}")
                        continue
                    except ValueError:
                        pass  # Good, it's not numeric
                
                # Ensure keyword is a meaningful string
                if not isinstance(keyword, str) or len(keyword.strip()) < 2:
                    logger.debug(f"🔸 YAKE: Skipping invalid keyword: {keyword}")
                    continue
                
                # YAKE scores are lower=better, so invert
                # Also ensure score is a number, not a string
                if isinstance(score, str):
                    try:
                        score = float(score)
                    except ValueError:
                        score = 1.0  # Default score if conversion fails
                        
                normalized_score = 1.0 / (1.0 + score)
                
                result.append({
                    'keyword': keyword,
                    'score': normalized_score,
                    'method': 'yake',
                    'language': language,
                    'weight': base_weight,
                    'raw_score': score
                })
                
            logger.debug(f"🔸 YAKE extracted {len(result)} keywords")
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ YAKE extraction failed: {e}")
            return []
    
    def _score_and_rank_keywords(self, extraction_results: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Score and rank keywords from multiple extraction methods"""
        if not extraction_results:
            return []
            
        # Combine scores for duplicate keywords
        keyword_scores = {}
        
        for result in extraction_results:
            # Ensure keyword is a string
            keyword_raw = result.get('keyword', '')
            if not isinstance(keyword_raw, str):
                keyword_raw = str(keyword_raw)
            
            # Validate keyword is not purely numeric
            if isinstance(keyword_raw, str):
                try:
                    float(keyword_raw.strip())
                    logger.debug(f"🔸 Scoring: Skipping numeric keyword: {keyword_raw}")
                    continue
                except ValueError:
                    pass  # Good, it's not numeric
                
            keyword = keyword_raw.lower().strip()
            if not keyword or len(keyword) < 2:
                continue
                
            if keyword not in keyword_scores:
                keyword_scores[keyword] = {
                    'keyword': keyword_raw,  # Keep original case
                    'total_score': 0.0,
                    'methods': [],
                    'count': 0
                }
            
            # Calculate weighted score - ensure score is a number
            score = result.get('score', 0.0)
            if not isinstance(score, (int, float)):
                try:
                    score = float(score)
                except (ValueError, TypeError):
                    score = 0.0
                    
            weight = result.get('weight', 1.0)
            if not isinstance(weight, (int, float)):
                try:
                    weight = float(weight)
                except (ValueError, TypeError):
                    weight = 1.0
                    
            weighted_score = score * weight
            keyword_scores[keyword]['total_score'] += weighted_score
            keyword_scores[keyword]['methods'].append(result.get('method', 'unknown'))
            keyword_scores[keyword]['count'] += 1
        
        # Apply domain validation boost
        if self.extraction_config.get('domain_validation', {}).get('enabled', True):
            self._apply_domain_boost(keyword_scores, text)
        
        # Convert to list and sort by score
        scored_keywords = []
        for keyword_data in keyword_scores.values():
            scored_keywords.append({
                'keyword': keyword_data['keyword'],
                'score': keyword_data['total_score'] / keyword_data['count'],  # Average score
                'total_score': keyword_data['total_score'],
                'methods': list(set(keyword_data['methods'])),  # Unique methods
                'method_count': len(set(keyword_data['methods']))
            })
        
        # Sort by total score descending
        scored_keywords.sort(key=lambda x: x['total_score'], reverse=True)
        
        return scored_keywords
    
    def _apply_domain_boost(self, keyword_scores: Dict[str, Dict], text: str):
        """Apply domain-specific validation and boosting"""
        domain_config = self.extraction_config.get('domain_validation', {})
        smt_terms = domain_config.get('smt_terms', [])
        boost_score = domain_config.get('boost_domain_score', 0.1)
        
        for keyword, data in keyword_scores.items():
            # Check if keyword contains domain terms
            for term in smt_terms:
                if term.lower() in keyword.lower():
                    data['total_score'] += boost_score
                    logger.debug(f"🔧 Applied domain boost to: {keyword}")
                    break
    
    def _combine_weighted_keywords(self, source_keywords: Dict[str, Dict], total_weight: float) -> List[Dict[str, Any]]:
        """Combine keywords from multiple sources with weighted scoring"""
        combined_keywords = {}
        
        for source_name, source_data in source_keywords.items():
            source_weight = source_data['weight'] / total_weight  # Normalize weight
            keywords = source_data['keywords']
            
            for keyword_data in keywords:
                keyword = keyword_data['keyword'].lower().strip()
                
                if keyword not in combined_keywords:
                    combined_keywords[keyword] = {
                        'keyword': keyword_data['keyword'],  # Keep original case
                        'total_score': 0.0,
                        'sources': [],
                        'methods': set()
                    }
                
                # Add weighted score from this source
                source_score = keyword_data['total_score'] * source_weight
                combined_keywords[keyword]['total_score'] += source_score
                combined_keywords[keyword]['sources'].append({
                    'source': source_name,
                    'score': keyword_data['total_score'],
                    'weight': source_weight
                })
                combined_keywords[keyword]['methods'].update(keyword_data['methods'])
        
        # Convert to list format
        result = []
        for keyword_data in combined_keywords.values():
            result.append({
                'keyword': keyword_data['keyword'],
                'total_score': keyword_data['total_score'],
                'sources': keyword_data['sources'],
                'methods': list(keyword_data['methods']),
                'source_count': len(keyword_data['sources'])
            })
        
        # Sort by total score
        result.sort(key=lambda x: x['total_score'], reverse=True)
        return result
    
    def _post_process_keywords(self, combined_keywords: List[Dict[str, Any]], 
                              content_context: Optional[Dict[str, Any]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """Post-process and filter final keywords with detailed metadata"""
        output_config = self.extraction_config.get('output', {})
        max_keywords = output_config.get('max_keywords', 5)
        min_threshold = output_config.get('min_score_threshold', 0.2)
        deduplicate = output_config.get('deduplicate', True)
        
        # Initialize post-processing metadata
        post_process_metadata = {
            'original_count': len(combined_keywords),
            'after_score_filter': 0,
            'after_deduplication': 0,
            'final_count': 0,
            'deduplication_stats': {
                'exact_duplicates_removed': 0,
                'substring_duplicates_removed': 0,
                'linguistic_variations_removed': 0,
                'total_removed': 0
            },
            'filter_stages': []
        }
        
        logger.debug(f"🔄 Starting post-processing with {len(combined_keywords)} keywords")
        
        # Stage 1: Filter by minimum score
        filtered_keywords = [
            kw for kw in combined_keywords 
            if kw['total_score'] >= min_threshold
        ]
        post_process_metadata['after_score_filter'] = len(filtered_keywords)
        post_process_metadata['filter_stages'].append({
            'stage': 'score_filter',
            'removed': len(combined_keywords) - len(filtered_keywords),
            'remaining': len(filtered_keywords)
        })
        
        logger.debug(f"📊 After score filter (>= {min_threshold}): {len(filtered_keywords)} keywords")
        
        # Stage 2: Enhanced deduplication if enabled
        if deduplicate:
            filtered_keywords, dedup_stats = self._advanced_deduplicate_keywords(
                filtered_keywords, content_context=content_context
            )
            post_process_metadata['deduplication_stats'] = dedup_stats
            post_process_metadata['after_deduplication'] = len(filtered_keywords)
            post_process_metadata['filter_stages'].append({
                'stage': 'deduplication',
                'removed': dedup_stats['total_removed'],
                'remaining': len(filtered_keywords)
            })
            
            logger.debug(f"🔗 After deduplication: {len(filtered_keywords)} keywords (removed {dedup_stats['total_removed']})")
        else:
            post_process_metadata['after_deduplication'] = len(filtered_keywords)
        
        # Stage 3: Take top N keywords
        top_keywords = filtered_keywords[:max_keywords]
        post_process_metadata['final_count'] = len(top_keywords)
        post_process_metadata['filter_stages'].append({
            'stage': 'top_n_selection',
            'removed': len(filtered_keywords) - len(top_keywords),
            'remaining': len(top_keywords)
        })
        
        logger.info(f"✅ Post-processing complete: {post_process_metadata['original_count']} → {post_process_metadata['final_count']} keywords")
        
        # Return keyword strings and metadata
        return [kw['keyword'] for kw in top_keywords], post_process_metadata
    
    def _advanced_deduplicate_keywords(self, filtered_keywords: List[Dict[str, Any]], 
                                      content_context: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Advanced deduplication using multiple configurable strategies
        
        Strategies applied in order:
        1. Remove exact duplicates (case-insensitive) 
        2. Remove substring duplicates (configurable)
        3. Remove linguistic variations (configurable)
        4. Remove semantic duplicates (configurable)
        """
        if not filtered_keywords:
            return [], {
                'total_removed': 0,
                'exact_duplicates_removed': 0,
                'substring_duplicates_removed': 0,
                'linguistic_variations_removed': 0,
                'semantic_duplicates_removed': 0,
                'strategies_applied': []
            }
        
        # Get advanced deduplication configuration
        advanced_dedup_config = self.extraction_config.get('output', {}).get('advanced_dedup', {})
        enabled = advanced_dedup_config.get('enabled', True)
        
        if not enabled:
            logger.debug("🔧 Advanced deduplication disabled in config")
            return filtered_keywords, {
                'total_removed': 0,
                'exact_duplicates_removed': 0,
                'substring_duplicates_removed': 0,
                'linguistic_variations_removed': 0,
                'semantic_duplicates_removed': 0,
                'strategies_applied': []
            }
        
        original_count = len(filtered_keywords)
        unique_keywords = filtered_keywords.copy()
        strategies_applied = []
        dedup_stats = {}
        
        # Strategy 1: Remove exact duplicates (always enabled)
        before_exact = len(unique_keywords)
        unique_keywords = self._remove_exact_duplicates(unique_keywords)
        exact_removed = before_exact - len(unique_keywords)
        dedup_stats['exact_duplicates_removed'] = exact_removed
        strategies_applied.append('exact_duplicates')
        
        # Strategy 2: Remove substring duplicates (configurable)
        if advanced_dedup_config.get('remove_substrings', True):
            before_substring = len(unique_keywords)
            unique_keywords = self._remove_substring_duplicates(unique_keywords)
            substring_removed = before_substring - len(unique_keywords)
            dedup_stats['substring_duplicates_removed'] = substring_removed
            strategies_applied.append('substring_duplicates')
        else:
            dedup_stats['substring_duplicates_removed'] = 0
        
        # Strategy 3: Remove linguistic variations (configurable)
        if advanced_dedup_config.get('remove_linguistic_variations', True):
            before_linguistic = len(unique_keywords)
            unique_keywords = self._remove_linguistic_variations(unique_keywords)
            linguistic_removed = before_linguistic - len(unique_keywords)
            dedup_stats['linguistic_variations_removed'] = linguistic_removed
            strategies_applied.append('linguistic_variations')
        else:
            dedup_stats['linguistic_variations_removed'] = 0
        
        # Strategy 4: Remove semantic duplicates (NEW - configurable)
        if advanced_dedup_config.get('semantic_similarity', False):
            before_semantic = len(unique_keywords)
            similarity_threshold = advanced_dedup_config.get('similarity_threshold', 0.85)
            preserve_domain_terms = advanced_dedup_config.get('preserve_domain_terms', True)
            unique_keywords, semantic_metadata = self._remove_semantic_duplicates(
                unique_keywords, 
                similarity_threshold=similarity_threshold,
                preserve_domain_terms=preserve_domain_terms,
                content_context=content_context
            )
            semantic_removed = before_semantic - len(unique_keywords)
            dedup_stats['semantic_duplicates_removed'] = semantic_removed
            dedup_stats['semantic_deduplication_metadata'] = semantic_metadata
            strategies_applied.append('semantic_duplicates')
        else:
            dedup_stats['semantic_duplicates_removed'] = 0
        
        # ADVANCED ENHANCEMENT: Enhanced domain validation
        if content_context:
            before_domain = len(unique_keywords)
            unique_keywords = self._enhanced_domain_validation(unique_keywords, content_context)
            domain_enhanced = len(unique_keywords) - before_domain  # This could be positive if new keywords added
            dedup_stats['domain_validation_applied'] = True
            dedup_stats['domain_enhanced_count'] = domain_enhanced
            strategies_applied.append('enhanced_domain_validation')
        else:
            dedup_stats['domain_validation_applied'] = False
        
        # Calculate total statistics
        total_removed = original_count - len(unique_keywords)
        dedup_stats.update({
            'total_removed': total_removed,
            'strategies_applied': strategies_applied,
            'original_count': original_count,
            'final_count': len(unique_keywords)
        })
        
        logger.debug(f"🔧 Advanced deduplication: {original_count} → {len(unique_keywords)} keywords "
                    f"(removed {total_removed}, strategies: {strategies_applied})")
        
        return unique_keywords, dedup_stats
    
    def _remove_exact_duplicates(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove exact case-insensitive duplicates"""
        seen = set()
        unique_keywords = []
        for kw in keywords:
            keyword_lower = kw['keyword'].lower().strip()
            if keyword_lower not in seen:
                seen.add(keyword_lower)
                unique_keywords.append(kw)
        return unique_keywords
    
    def _remove_substring_duplicates(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove shorter keywords that are substrings of longer ones"""
        # Sort by length (longest first) and score (highest first)
        sorted_keywords = sorted(keywords, key=lambda x: (-len(x['keyword']), -x['total_score']))
        
        unique_keywords = []
        seen_patterns = []
        
        for kw in sorted_keywords:
            keyword_clean = self._clean_keyword(kw['keyword'])
            
            # Check if this keyword is a substring of any already accepted keyword
            is_substring = False
            for i, existing_pattern in enumerate(seen_patterns):
                if keyword_clean in existing_pattern:
                    # This keyword is contained in a longer one, skip it
                    is_substring = True
                    break
                elif existing_pattern in keyword_clean:
                    # This keyword contains a shorter one, replace the shorter one
                    seen_patterns[i] = keyword_clean
                    # Remove the shorter keyword from unique_keywords
                    unique_keywords = [k for k in unique_keywords if self._clean_keyword(k['keyword']) != existing_pattern]
                    break
            
            if not is_substring:
                seen_patterns.append(keyword_clean)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def _remove_linguistic_variations(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove linguistic variations like plural/singular, articles"""
        unique_keywords = []
        seen_roots = set()
        
        for kw in keywords:
            keyword = kw['keyword']
            normalized = self._normalize_keyword(keyword)
            
            if normalized not in seen_roots:
                seen_roots.add(normalized)
                unique_keywords.append(kw)
        
        return unique_keywords
    
    def _normalize_keyword(self, keyword: str) -> str:
        """Normalize keyword for comparison"""
        import re
        
        # Convert to lowercase
        normalized = keyword.lower().strip()
        
        # Remove common articles (English and Chinese)
        normalized = re.sub(r'\b(the|a|an|这|那|一个)\b\s*', '', normalized)
        
        # Handle plural/singular for English
        # Simple approach: remove trailing 's' if word is longer than 3 chars
        words = normalized.split()
        normalized_words = []
        for word in words:
            if len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
                # Check if it's likely a plural (not words like "glass", "class")
                if not word.endswith(('us', 'is', 'ss', 'ous')):
                    singular = word[:-1]
                    normalized_words.append(singular)
                else:
                    normalized_words.append(word)
            else:
                normalized_words.append(word)
        
        normalized = ' '.join(normalized_words)
        
        # Remove extra whitespace and punctuation
        normalized = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _remove_semantic_duplicates(self, keywords: List[Dict[str, Any]], 
                                   similarity_threshold: float = 0.85,
                                   preserve_domain_terms: bool = True,
                                   content_context: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Remove semantically similar keywords using intelligent model selection
        
        Args:
            keywords: List of keyword dictionaries
            similarity_threshold: Base cosine similarity threshold (0-1, higher = more strict)
            preserve_domain_terms: Keep domain-specific terms even if similar
            content_context: Context for intelligent processing decisions
            
        Returns:
            Tuple of (filtered_keywords, metadata)
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("⚠️ sklearn not available, skipping semantic deduplication")
            return keywords, {"method_used": "none", "reason": "sklearn_unavailable"}
        
        if len(keywords) <= 1:
            return keywords, {"method_used": "none", "reason": "insufficient_keywords"}
        
        start_time = time.time()
        self.performance_monitor.record_memory_usage()
        
        # Get semantic models configuration from testset_generation config
        advanced_dedup_config = self.extraction_config.get('output', {}).get('advanced_dedup', {})
        semantic_models_config = advanced_dedup_config.get('semantic_models', {})
        semantic_strategy_config = advanced_dedup_config.get('semantic_strategy', {})
        
        # Configure similarity thresholds based on language and strategy
        method = semantic_strategy_config.get('method', 'adaptive_language')
        language_detection_enabled = semantic_strategy_config.get('language_detection', True)
        
        # ADVANCED ENHANCEMENT: Content-aware analysis
        if content_context is None:
            content_context = {
                'complexity_score': 0.5,
                'keyword_density': 0.1,
                'domain_specificity': 0.5,
                'total_content_length': 1000,
                'language_mixing_score': 0.0
            }
        
        # ADVANCED ENHANCEMENT: Context-aware model selection
        if method == 'adaptive_strategy':
            # Use intelligent strategy selection
            optimal_method = self._adaptive_strategy_selection(content_context)
            logger.info(f"🧠 Intelligent strategy selection: {method} → {optimal_method}")
            method = optimal_method
        
        # Model selection with health checking
        if method == 'context_aware':
            model_to_use = self._context_aware_selection(semantic_models_config, content_context)
        else:
            model_to_use = self._select_semantic_model(semantic_models_config, semantic_strategy_config, keywords)
        
        if not model_to_use:
            logger.warning("⚠️ No suitable semantic model found, skipping semantic deduplication")
            return keywords, {"method_used": "none", "reason": "no_model_available"}
        
        # ADVANCED ENHANCEMENT: Model health check
        health_status = self._model_health_check(model_to_use)
        if not health_status['available'] or health_status['fallback_recommended']:
            logger.warning(f"⚠️ Model health issues detected: {health_status['issues']}")
            if 'fallback' in semantic_models_config:
                model_to_use = semantic_models_config['fallback']
                logger.info("🔄 Switching to fallback model")
            else:
                return keywords, {"method_used": "none", "reason": "model_health_failed"}
        
        model_name = model_to_use['model_name']
        logger.info(f"🔗 Using semantic model: {model_name} for deduplication")
        
        try:
            # Extract keyword texts for processing
            keyword_texts = [kw['keyword'] for kw in keywords]
            
            # ADVANCED ENHANCEMENT: Check embedding cache
            cached_embeddings = self.embedding_cache.get_embeddings(keyword_texts, model_name)
            
            if cached_embeddings is not None:
                embeddings = cached_embeddings
                logger.debug(f"✅ Using cached embeddings for {len(keyword_texts)} keywords")
                self.performance_monitor.session_stats['cache_hits'] += 1
            else:
                # ADVANCED ENHANCEMENT: Batch processing optimization
                batch_config = self._optimize_batch_processing(keyword_texts, model_to_use)
                optimal_batch_size = batch_config['optimal_batch_size']
                
                # Load the embedding model
                model = self._load_semantic_model(model_to_use)
                if not model:
                    return keywords, {"method_used": "none", "reason": "model_load_failed"}
                
                # Generate embeddings with optimized batching
                try:
                    if batch_config['processing_strategy'] == 'single_batch':
                        embeddings = model.encode(keyword_texts, convert_to_tensor=False, batch_size=optimal_batch_size)
                    else:
                        # Process in optimized batches
                        embeddings = []
                        for i in range(0, len(keyword_texts), optimal_batch_size):
                            batch = keyword_texts[i:i + optimal_batch_size]
                            batch_embeddings = model.encode(batch, convert_to_tensor=False)
                            embeddings.extend(batch_embeddings)
                    
                    # Cache the embeddings
                    self.embedding_cache.cache_embeddings(keyword_texts, model_name, embeddings)
                    self.performance_monitor.session_stats['cache_misses'] += 1
                    
                    logger.debug(f"✅ Generated embeddings for {len(keyword_texts)} keywords with batch_size={optimal_batch_size}")
                    
                except Exception as e:
                    logger.error(f"❌ Failed to generate embeddings: {e}")
                    return keywords, {"method_used": "none", "reason": "embedding_failed", "error": str(e)}
            
            # Detect language if enabled
            language_info = {}
            if language_detection_enabled:
                language_info = self._detect_keywords_language(keyword_texts)
            
            # ADVANCED ENHANCEMENT: Advanced threshold adjustment
            adjusted_threshold = self._advanced_threshold_adjustment(
                similarity_threshold, content_context
            )
            
            # Also apply basic language adjustment
            final_threshold = self._get_adjusted_threshold(
                adjusted_threshold, language_info, semantic_strategy_config
            )
            
            # Calculate cosine similarity matrix
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(embeddings)
            
            # Find duplicate pairs above threshold
            duplicates_found = []
            keywords_to_remove = set()
            
            for i in range(len(keywords)):
                if i in keywords_to_remove:
                    continue
                    
                for j in range(i + 1, len(keywords)):
                    if j in keywords_to_remove:
                        continue
                        
                    similarity_score = similarity_matrix[i][j]
                    
                    if similarity_score >= final_threshold:
                        # Decide which keyword to keep
                        keep_idx, remove_idx = self._decide_keyword_to_keep(
                            keywords[i], keywords[j], i, j, preserve_domain_terms
                        )
                        
                        duplicates_found.append({
                            'keyword_1': keywords[i]['keyword'],
                            'keyword_2': keywords[j]['keyword'],
                            'similarity_score': float(similarity_score),
                            'kept': keywords[keep_idx]['keyword'],
                            'removed': keywords[remove_idx]['keyword']
                        })
                        
                        keywords_to_remove.add(remove_idx)
            
            # Filter out removed keywords
            filtered_keywords = [kw for i, kw in enumerate(keywords) if i not in keywords_to_remove]
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Record performance monitoring data
            self.performance_monitor.record_model_selection(
                model_name,
                f"semantic_deduplication_{method}",
                processing_time / 1000,  # Convert back to seconds for monitoring
                len(keywords_to_remove) > 0  # Success if duplicates were removed
            )
            
            # Clean up embedding cache periodically
            if self.performance_monitor.session_stats['cache_misses'] > 10:
                self.embedding_cache.cleanup_cache()
            
            # Get performance summary for metadata
            performance_summary = self.performance_monitor.get_performance_summary()
            
            metadata = {
                "method_used": "sentence_transformers",
                "model_used": model_name,
                "similarity_threshold": final_threshold,
                "original_threshold": similarity_threshold,
                "advanced_threshold": adjusted_threshold,
                "language_detection": language_info,
                "strategy_used": method,
                "pairs_compared": len(keyword_texts) * (len(keyword_texts) - 1) // 2,
                "duplicates_found": duplicates_found,
                "keywords_removed": len(keywords_to_remove),
                "keywords_kept": len(filtered_keywords),
                "processing_time_ms": round(processing_time, 2),
                "performance_summary": performance_summary,
                "cache_stats": self.embedding_cache.cache_stats.copy(),
                "content_context_used": content_context is not None,
                "advanced_features_used": {
                    "advanced_threshold_adjustment": True,
                    "model_health_check": health_status,
                    "batch_optimization": batch_config if 'batch_config' in locals() else None,
                    "embedding_cache": cached_embeddings is not None
                }
            }
            
            logger.info(f"🔗 Semantic deduplication: {len(keywords)} → {len(filtered_keywords)} keywords "
                       f"({len(keywords_to_remove)} removed, threshold: {final_threshold:.3f}, "
                       f"cache_hit_rate: {performance_summary['cache_hit_rate']:.2f})")
            
            return filtered_keywords, metadata
            
        except Exception as e:
            logger.error(f"❌ Semantic deduplication failed: {e}")
            return keywords, {"method_used": "failed", "error": str(e)}
    
    
    def _select_semantic_model(self, models_config: Dict[str, Any], strategy_config: Dict[str, Any], 
                              keywords: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best semantic model based on strategy and keyword content"""
        method = strategy_config.get('method', 'adaptive_language')
        
        if method == 'adaptive_language':
            # Detect dominant language in keywords
            keyword_texts = [kw['keyword'] for kw in keywords]
            combined_text = " ".join(keyword_texts)
            
            # Simple language detection based on character sets
            chinese_chars = len([c for c in combined_text if '\u4e00' <= c <= '\u9fff'])
            total_chars = len([c for c in combined_text if c.isalpha()])
            
            if total_chars == 0:
                return models_config.get('primary', models_config.get('fallback'))
            
            chinese_ratio = chinese_chars / total_chars
            
            if chinese_ratio > 0.3:  # Significant Chinese content
                return models_config.get('chinese_specific') or models_config.get('primary')
            elif chinese_ratio < 0.1:  # Predominantly English
                return models_config.get('english_specific') or models_config.get('primary')
            else:  # Mixed content
                return models_config.get('primary')
        
        elif method == 'best_available':
            # Try models in order of preference
            for model_key in ['primary', 'chinese_specific', 'english_specific', 'fallback']:
                if model_key in models_config and models_config[model_key].get('enabled', True):
                    return models_config[model_key]
        
        elif method == 'specific_model':
            # Use a specific model if configured
            specific_model = strategy_config.get('specific_model_key', 'primary')
            return models_config.get(specific_model)
        
        # Fallback
        return models_config.get('fallback')
    
    def _load_semantic_model(self, model_config: Dict[str, Any]):
        """Load semantic model with configuration"""
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = model_config.get('model_name')
            cache_dir = model_config.get('cache_dir', './cache/sentence_transformers')
            offline_mode = model_config.get('offline_mode', True)
            local_files_only = model_config.get('local_files_only', True)
            device = model_config.get('device', 'cpu')
            
            logger.debug(f"🔧 Loading semantic model: {model_name}")
            
            model = SentenceTransformer(
                model_name,
                cache_folder=cache_dir,
                device=device,
                use_auth_token=False
            )
            
            if offline_mode:
                # Configure for offline use
                model.max_seq_length = 512
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Failed to load semantic model {model_config.get('model_name')}: {e}")
            
            # Try fallback models if available
            fallback_enabled = model_config.get('fallback_enabled', False)
            if fallback_enabled:
                try:
                    from sentence_transformers import SentenceTransformer
                    fallback_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
                    logger.info("✅ Loaded fallback model: all-MiniLM-L6-v2")
                    return fallback_model
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback model also failed: {fallback_error}")
            
            return None
    
    def _detect_keywords_language(self, keyword_texts: List[str]) -> Dict[str, Any]:
        """Detect language distribution in keywords"""
        combined_text = " ".join(keyword_texts)
        
        chinese_chars = len([c for c in combined_text if '\u4e00' <= c <= '\u9fff'])
        english_chars = len([c for c in combined_text if c.isalpha() and not ('\u4e00' <= c <= '\u9fff')])
        total_chars = chinese_chars + english_chars
        
        if total_chars == 0:
            return {"language": "unknown", "chinese_ratio": 0, "english_ratio": 0}
        
        chinese_ratio = chinese_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if chinese_ratio > 0.6:
            language = "chinese"
        elif english_ratio > 0.6:
            language = "english"
        else:
            language = "mixed"
        
        return {
            "language": language,
            "chinese_ratio": chinese_ratio,
            "english_ratio": english_ratio,
            "total_chars": total_chars
        }
    
    def _get_adjusted_threshold(self, base_threshold: float, language_info: Dict[str, Any], 
                               strategy_config: Dict[str, Any]) -> float:
        """Adjust similarity threshold based on language"""
        language = language_info.get('language', 'unknown')
        
        if language == 'chinese':
            return strategy_config.get('chinese_threshold', base_threshold)
        elif language == 'english':
            return strategy_config.get('english_threshold', base_threshold)
        elif language == 'mixed':
            return strategy_config.get('mixed_language_threshold', base_threshold)
        else:
            return base_threshold
    
    def _decide_keyword_to_keep(self, kw1: Dict[str, Any], kw2: Dict[str, Any], 
                               idx1: int, idx2: int, preserve_domain_terms: bool) -> Tuple[int, int]:
        """Decide which keyword to keep when removing semantic duplicates"""
        
        # Priority 1: Preserve domain terms
        if preserve_domain_terms:
            kw1_is_domain = self._is_domain_term(kw1['keyword'])
            kw2_is_domain = self._is_domain_term(kw2['keyword'])
            
            if kw1_is_domain and not kw2_is_domain:
                return idx1, idx2  # Keep kw1
            elif kw2_is_domain and not kw1_is_domain:
                return idx2, idx1  # Keep kw2
        
        # Priority 2: Keep higher scored keyword
        score1 = kw1.get('score', 0)
        score2 = kw2.get('score', 0)
        
        if score1 > score2:
            return idx1, idx2
        elif score2 > score1:
            return idx2, idx1
        
        # Priority 3: Keep longer keyword (more specific)
        if len(kw1['keyword']) > len(kw2['keyword']):
            return idx1, idx2
        elif len(kw2['keyword']) > len(kw1['keyword']):
            return idx2, idx1
        
        # Priority 4: Keep first keyword (arbitrary but consistent)
        return idx1, idx2

    def _is_domain_term(self, keyword: str) -> bool:
        """
        Check if a keyword is likely a domain-specific technical term
        
        Heuristics:
        - Contains technical prefixes/suffixes
        - Has mixed case (CamelCase, technical terms)
        - Contains numbers or special characters
        - Is longer than average (compound technical terms)
        - Contains Chinese characters (for this domain)
        """
        if not keyword:
            return False
        
        keyword = keyword.strip()
        
        # Technical indicators
        technical_patterns = [
            # Technical suffixes/prefixes
            r'\b(meta|auto|semi|multi|inter|intra|pre|post|pro|anti)\w+',
            r'\w+(tion|sion|ment|ness|able|ible|ical|ical|ous|ive)',
            # Mixed case (technical terms)
            r'[a-z][A-Z]',
            # Contains numbers
            r'\d',
            # Technical punctuation
            r'[_\-\.]+',
            # Chinese characters (domain-specific)
            r'[\u4e00-\u9fff]'
        ]
        
        for pattern in technical_patterns:
            if re.search(pattern, keyword):
                return True
        
        # Length-based heuristic (longer compound terms are often technical)
        if len(keyword) > 15 or len(keyword.split()) > 2:
            return True
        
        return False
    
    def _clean_keyword(self, keyword: str) -> str:
        """Clean keyword for substring comparison"""
        import re
        # Remove punctuation and extra spaces, keep Chinese characters
        cleaned = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', keyword.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    def extract_simple(self, text: str, max_keywords: int = 5) -> str:
        """
        Simple extraction method for backward compatibility
        Returns comma-separated keyword string
        """
        if not text.strip():
            return ""
            
        # Use the hybrid extraction on the single text
        keywords = self._extract_keywords_from_text(text)
        
        # Filter and limit
        output_config = self.extraction_config.get('output', {})
        min_threshold = output_config.get('min_score_threshold', 0.2)
        
        filtered_keywords = [
            kw['keyword'] for kw in keywords 
            if kw.get('total_score', kw.get('score', 0)) >= min_threshold
        ]
        
        # Return top keywords as comma-separated string
        return ', '.join(filtered_keywords[:max_keywords])

    def extract_keywords_with_metadata(self, user_query: str, reference_contexts: List[str], reference_answer: str, sample_id: int = 0) -> Tuple[str, Dict[str, Any]]:
        """
        Extract keywords and return both keyword string and detailed metadata
        
        Args:
            user_query: The question/query text
            reference_contexts: List of context documents
            reference_answer: Generated answer text
            sample_id: Sample identifier for tracking
            
        Returns:
            Tuple of (keyword_string, metadata_dict)
        """
        try:
            # Extract with full metadata
            result = self.extract_keywords_from_sources(
                user_query=user_query,
                reference_contexts=reference_contexts,
                reference_answer=reference_answer
            )
            
            # Prepare comprehensive metadata
            metadata = {
                'sample_id': sample_id,
                'user_query': user_query or "",
                'keywords': result['keywords'],
                'source_breakdown': result['source_breakdown'],
                'language_detection': result['language_detection'],
                'language_distribution': result['language_distribution'],
                'keyword_details': result['keyword_details'],
                'extraction_metadata': result['extraction_metadata'],
                'post_process_metadata': result.get('post_process_metadata', {}),
                'extraction_config': {
                    'source_weights': self.source_weights,
                    'method': result['extraction_metadata']['method'],
                    'models_used': {
                        'keybert_models': list(self.models.get('keybert', {}).keys()) if 'keybert' in self.models else [],
                        'spacy_models': list(self.models.get('spacy', {}).keys()) if 'spacy' in self.models else [],
                        'yake_available': YAKE_AVAILABLE
                    }
                }
            }
            
            keyword_string = ', '.join(result['keywords'])
            return keyword_string, metadata
            
        except Exception as e:
            logger.warning(f"⚠️ Enhanced keyword extraction failed: {e}")
            return "", {
                'sample_id': sample_id,
                'error': str(e),
                'keywords': [],
                'extraction_failed': True
            }

    # =====================================
    # ADVANCED INTELLIGENT MODEL SELECTION
    # =====================================
    
    def _advanced_threshold_adjustment(self, base_threshold: float, context: Dict[str, Any]) -> float:
        """
        Advanced threshold adjustment with multiple factors
        
        Considers:
        - Document complexity (vocabulary diversity)
        - Keyword density
        - Domain specificity
        - Content length
        - Language mixing
        """
        adjusted_threshold = base_threshold
        
        # Factor 1: Document complexity
        complexity_score = context.get('complexity_score', 0.5)
        if complexity_score > 0.7:  # High complexity
            adjusted_threshold += 0.03  # More strict for complex content
        elif complexity_score < 0.3:  # Simple content
            adjusted_threshold -= 0.02  # More lenient for simple content
        
        # Factor 2: Keyword density
        keyword_density = context.get('keyword_density', 0.1)
        if keyword_density > 0.2:  # High keyword density
            adjusted_threshold += 0.02  # More strict to avoid noise
        elif keyword_density < 0.05:  # Low keyword density
            adjusted_threshold -= 0.03  # More lenient to capture keywords
        
        # Factor 3: Domain specificity
        domain_score = context.get('domain_specificity', 0.5)
        if domain_score > 0.8:  # Highly domain-specific
            adjusted_threshold -= 0.02  # More lenient for domain terms
        
        # Factor 4: Content length adjustment
        content_length = context.get('total_content_length', 1000)
        if content_length > 5000:  # Long content
            adjusted_threshold += 0.01  # Slightly more strict
        elif content_length < 500:  # Short content
            adjusted_threshold -= 0.02  # More lenient
        
        # Factor 5: Language mixing penalty
        language_mixing = context.get('language_mixing_score', 0.0)
        if language_mixing > 0.3:  # High language mixing
            adjusted_threshold -= 0.02  # More lenient due to complexity
        
        # Ensure threshold stays within reasonable bounds
        adjusted_threshold = max(0.6, min(0.95, adjusted_threshold))
        
        logger.debug(f"🎯 Advanced threshold adjustment: {base_threshold:.3f} → {adjusted_threshold:.3f}")
        logger.debug(f"   Factors: complexity={complexity_score:.2f}, density={keyword_density:.2f}, "
                    f"domain={domain_score:.2f}, length={content_length}, mixing={language_mixing:.2f}")
        
        return adjusted_threshold
    
    def _context_aware_selection(self, models_config: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Select model based on document context and content characteristics
        
        Analyzes:
        - Document domain (technical, general, etc.)
        - Content complexity
        - Language distribution
        - Processing requirements
        """
        content_analysis = context.get('content_analysis', {})
        
        # Domain-based selection
        dominant_domain = content_analysis.get('dominant_domain', 'general')
        if dominant_domain == 'technical' and 'chinese_specific' in models_config:
            # Technical Chinese content benefits from specialized model
            logger.debug("🎯 Selected Chinese-specific model for technical domain")
            return models_config['chinese_specific']
        
        # Complexity-based selection
        complexity_score = content_analysis.get('complexity_score', 0.5)
        if complexity_score > 0.8:
            # High complexity requires best available model
            for model_key in ['primary', 'english_specific', 'chinese_specific']:
                if model_key in models_config and models_config[model_key].get('enabled', True):
                    logger.debug(f"🎯 Selected {model_key} model for high complexity content")
                    return models_config[model_key]
        
        # Language distribution based selection
        language_dist = content_analysis.get('language_distribution', {})
        chinese_ratio = language_dist.get('chinese', 0)
        english_ratio = language_dist.get('english', 0)
        
        if chinese_ratio > 0.7 and 'chinese_specific' in models_config:
            logger.debug("🎯 Selected Chinese-specific model for high Chinese ratio")
            return models_config['chinese_specific']
        elif english_ratio > 0.7 and 'english_specific' in models_config:
            logger.debug("🎯 Selected English-specific model for high English ratio")
            return models_config['english_specific']
        
        # Default to primary model
        logger.debug("🎯 Selected primary model as default")
        return models_config.get('primary', models_config.get('fallback'))
    
    def _model_health_check(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check model availability and performance status
        
        Returns health status including:
        - Availability
        - Performance metrics
        - Memory requirements
        - Fallback recommendations
        """
        health_status = {
            'available': False,
            'performance_score': 0.0,
            'memory_efficient': True,
            'load_time_estimate': 0.0,
            'fallback_recommended': False,
            'issues': []
        }
        
        model_name = model_config.get('model_name', '')
        
        try:
            # Test basic model loading
            start_time = time.time()
            if KEYBERT_AVAILABLE:
                # Quick availability check
                test_model = SentenceTransformer(
                    model_name,
                    cache_folder=model_config.get('cache_dir', './cache/sentence_transformers'),
                    device='cpu'  # Use CPU for health check
                )
                load_time = time.time() - start_time
                
                health_status.update({
                    'available': True,
                    'load_time_estimate': load_time,
                    'performance_score': max(0.1, min(1.0, 1.0 - (load_time / 30.0)))  # Score based on load time
                })
                
                # Memory check
                if PSUTIL_AVAILABLE:
                    process = psutil.Process()
                    memory_before = process.memory_info().rss
                    
                    # Test small encoding
                    test_model.encode(["test"], convert_to_tensor=False)
                    
                    memory_after = process.memory_info().rss
                    memory_used_mb = (memory_after - memory_before) / 1024 / 1024
                    
                    health_status['memory_efficient'] = memory_used_mb < 500  # Less than 500MB
                    
                    if memory_used_mb > 1000:  # More than 1GB
                        health_status['issues'].append('high_memory_usage')
                        health_status['fallback_recommended'] = True
                
                del test_model  # Clean up
                
            else:
                health_status['issues'].append('keybert_not_available')
                health_status['fallback_recommended'] = True
                
        except Exception as e:
            health_status['issues'].append(f'load_error: {str(e)}')
            health_status['fallback_recommended'] = True
            logger.debug(f"Model health check failed for {model_name}: {e}")
        
        logger.debug(f"🏥 Model health check for {model_name}: score={health_status['performance_score']:.2f}, "
                    f"available={health_status['available']}, issues={health_status['issues']}")
        
        return health_status
    
    def _optimize_batch_processing(self, keywords: List[str], model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize batch size and processing strategy based on:
        - Available memory
        - Model requirements
        - Keyword count
        - System resources
        """
        optimization = {
            'optimal_batch_size': 32,
            'processing_strategy': 'batch',
            'memory_allocation': 'normal',
            'parallel_processing': False
        }
        
        keyword_count = len(keywords)
        
        # Adjust batch size based on keyword count
        if keyword_count <= 10:
            optimization['optimal_batch_size'] = keyword_count
            optimization['processing_strategy'] = 'single_batch'
        elif keyword_count <= 50:
            optimization['optimal_batch_size'] = min(16, keyword_count)
        elif keyword_count <= 200:
            optimization['optimal_batch_size'] = 32
        else:
            optimization['optimal_batch_size'] = 64
            optimization['processing_strategy'] = 'large_batch'
        
        # Memory-based adjustments
        if PSUTIL_AVAILABLE:
            try:
                available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
                
                if available_memory < 2:  # Less than 2GB
                    optimization['optimal_batch_size'] = min(8, optimization['optimal_batch_size'])
                    optimization['memory_allocation'] = 'conservative'
                elif available_memory > 8:  # More than 8GB
                    optimization['parallel_processing'] = True
                    optimization['memory_allocation'] = 'aggressive'
                    
            except Exception as e:
                logger.debug(f"Failed to check memory for batch optimization: {e}")
        
        # Model-specific adjustments
        model_name = model_config.get('model_name', '')
        if 'large' in model_name.lower() or 'xl' in model_name.lower():
            optimization['optimal_batch_size'] = max(4, optimization['optimal_batch_size'] // 2)
            optimization['memory_allocation'] = 'conservative'
        
        logger.debug(f"🚀 Batch optimization: {keyword_count} keywords → "
                    f"batch_size={optimization['optimal_batch_size']}, "
                    f"strategy={optimization['processing_strategy']}")
        
        return optimization
    
    def _enhanced_domain_validation(self, keywords: List[Dict[str, Any]], domain_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Enhanced domain-specific keyword validation and boosting
        
        Features:
        - Dynamic domain term detection
        - Context-aware domain validation
        - Adaptive scoring based on domain relevance
        """
        if not keywords:
            return keywords
        
        domain_config = self.extraction_config.get('output', {}).get('domain_validation', {})
        if not domain_config.get('enabled', False):
            return keywords
        
        # Extract domain context
        content_domain = domain_context.get('dominant_domain', 'general')
        domain_terms = set(domain_config.get('smt_terms', []))
        boost_score = domain_config.get('boost_domain_score', 0.1)
        
        # Dynamic domain term expansion based on content
        dynamic_domain_terms = self._extract_dynamic_domain_terms(domain_context)
        domain_terms.update(dynamic_domain_terms)
        
        enhanced_keywords = []
        
        for kw_data in keywords:
            keyword = kw_data['keyword']
            current_score = kw_data.get('score', 0.5)
            
            # Domain relevance scoring
            domain_relevance = self._calculate_domain_relevance(keyword, domain_terms, content_domain)
            
            # Apply domain boosting
            if domain_relevance > 0.5:
                boosted_score = min(1.0, current_score + (boost_score * domain_relevance))
                kw_data['score'] = boosted_score
                kw_data['domain_boosted'] = True
                kw_data['domain_relevance'] = domain_relevance
                logger.debug(f"🎯 Domain boosted '{keyword}': {current_score:.3f} → {boosted_score:.3f}")
            
            enhanced_keywords.append(kw_data)
        
        # Sort by boosted scores
        enhanced_keywords.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return enhanced_keywords
    
    def _extract_dynamic_domain_terms(self, domain_context: Dict[str, Any]) -> set:
        """Extract domain-specific terms dynamically from content"""
        dynamic_terms = set()
        
        # Extract from content patterns
        content_analysis = domain_context.get('content_analysis', {})
        technical_terms = content_analysis.get('technical_terms', [])
        
        for term in technical_terms:
            if self._is_domain_term(term):
                dynamic_terms.add(term.lower())
        
        return dynamic_terms
    
    def _calculate_domain_relevance(self, keyword: str, domain_terms: set, content_domain: str) -> float:
        """Calculate how relevant a keyword is to the identified domain"""
        relevance_score = 0.0
        
        keyword_lower = keyword.lower()
        
        # Direct domain term match
        if keyword_lower in domain_terms:
            relevance_score += 0.8
        
        # Partial domain term match
        for domain_term in domain_terms:
            if domain_term in keyword_lower or keyword_lower in domain_term:
                relevance_score += 0.3
                break
        
        # Technical term indicators
        if self._is_domain_term(keyword):
            relevance_score += 0.4
        
        # Domain-specific patterns
        domain_patterns = self.domain_knowledge.get('domain_indicators', {})
        if content_domain in domain_patterns:
            for indicator in domain_patterns[content_domain]:
                if indicator.lower() in keyword_lower:
                    relevance_score += 0.2
                    break
        
        return min(1.0, relevance_score)
    
    def _adaptive_strategy_selection(self, content_analysis: Dict[str, Any]) -> str:
        """
        Dynamically select the best strategy based on content analysis
        
        Strategies:
        - adaptive_language: Best for mixed multilingual content
        - best_available: Best for general content with unknown characteristics  
        - specific_model: Best when domain/language is clearly identified
        """
        # Analyze content characteristics
        language_dist = content_analysis.get('language_distribution', {})
        complexity_score = content_analysis.get('complexity_score', 0.5)
        domain_specificity = content_analysis.get('domain_specificity', 0.5)
        
        chinese_ratio = language_dist.get('chinese', 0)
        english_ratio = language_dist.get('english', 0)
        
        # Decision logic
        if chinese_ratio > 0.8 or english_ratio > 0.8:
            # Clear language dominance
            strategy = 'specific_model'
            logger.debug(f"🎯 Selected specific_model strategy: Chinese={chinese_ratio:.2f}, English={english_ratio:.2f}")
        
        elif complexity_score > 0.7 and domain_specificity > 0.6:
            # Complex domain content
            strategy = 'best_available'
            logger.debug(f"🎯 Selected best_available strategy: complexity={complexity_score:.2f}, domain={domain_specificity:.2f}")
        
        elif abs(chinese_ratio - english_ratio) < 0.3:
            # Mixed language content
            strategy = 'adaptive_language'
            logger.debug(f"🎯 Selected adaptive_language strategy: mixed content detected")
        
        else:
            # Default adaptive approach
            strategy = 'adaptive_language'
            logger.debug(f"🎯 Selected adaptive_language strategy: default choice")
        
        return strategy
    
    def _analyze_content_characteristics(self, user_query: str, reference_contexts: List[str], 
                                       reference_answer: str) -> Dict[str, Any]:
        """
        Comprehensive content analysis for intelligent model selection
        
        Returns analysis including:
        - Language distribution
        - Complexity metrics
        - Domain classification
        - Technical term density
        """
        # Combine all content
        all_content = []
        if user_query:
            all_content.append(user_query)
        if reference_contexts:
            all_content.extend(reference_contexts)
        if reference_answer:
            all_content.append(reference_answer)
        
        combined_text = " ".join(all_content)
        
        # Generate cache key
        content_hash = hashlib.md5(combined_text.encode('utf-8')).hexdigest()[:16]
        
        # Check cache
        if content_hash in self.content_analysis_cache:
            return self.content_analysis_cache[content_hash]
        
        analysis = {
            'total_content_length': len(combined_text),
            'language_distribution': self._analyze_language_distribution(combined_text),
            'complexity_score': self._calculate_complexity_score(combined_text),
            'domain_specificity': self._calculate_domain_specificity(combined_text),
            'technical_terms': self._extract_technical_terms(combined_text),
            'keyword_density': self._calculate_keyword_density(combined_text),
            'language_mixing_score': self._calculate_language_mixing(combined_text),
            'dominant_domain': self._classify_content_domain(combined_text)
        }
        
        # Cache result
        self.content_analysis_cache[content_hash] = analysis
        
        return analysis
    
    def _analyze_language_distribution(self, text: str) -> Dict[str, float]:
        """Analyze language distribution in text"""
        if not text:
            return {'chinese': 0.0, 'english': 0.0, 'other': 0.0}
        
        chinese_chars = len([c for c in text if '\u4e00' <= c <= '\u9fff'])
        english_chars = len([c for c in text if c.isalpha() and ord(c) < 128])
        other_chars = len([c for c in text if c.isalpha() and not ('\u4e00' <= c <= '\u9fff') and ord(c) >= 128])
        
        total_chars = chinese_chars + english_chars + other_chars
        
        if total_chars == 0:
            return {'chinese': 0.0, 'english': 1.0, 'other': 0.0}
        
        return {
            'chinese': chinese_chars / total_chars,
            'english': english_chars / total_chars,
            'other': other_chars / total_chars
        }
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate content complexity based on vocabulary diversity and structure"""
        if not text:
            return 0.0
        
        words = text.split()
        if len(words) < 10:
            return 0.3  # Short content is simple
        
        # Vocabulary diversity (unique words / total words)
        unique_words = len(set(word.lower() for word in words))
        vocabulary_diversity = unique_words / len(words)
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Sentence complexity (approximate)
        sentences = text.split('.')
        avg_sentence_length = len(words) / max(1, len(sentences))
        
        # Combine metrics
        complexity = (
            vocabulary_diversity * 0.4 +
            min(1.0, avg_word_length / 8.0) * 0.3 +
            min(1.0, avg_sentence_length / 20.0) * 0.3
        )
        
        return min(1.0, complexity)
    
    def _calculate_domain_specificity(self, text: str) -> float:
        """Calculate how domain-specific the content is"""
        if not text:
            return 0.0
        
        words = text.lower().split()
        technical_indicators = 0
        
        # Check for technical patterns
        for word in words:
            if self._is_domain_term(word):
                technical_indicators += 1
        
        # Domain specificity ratio
        domain_score = technical_indicators / max(1, len(words))
        
        return min(1.0, domain_score * 5)  # Amplify signal
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """Extract likely technical terms from text"""
        words = text.split()
        technical_terms = []
        
        for word in words:
            cleaned_word = re.sub(r'[^\w\u4e00-\u9fff]', '', word)
            if cleaned_word and self._is_domain_term(cleaned_word):
                technical_terms.append(cleaned_word)
        
        return list(set(technical_terms))  # Remove duplicates
    
    def _calculate_keyword_density(self, text: str) -> float:
        """Calculate the density of potential keywords in text"""
        if not text:
            return 0.0
        
        words = text.split()
        potential_keywords = 0
        
        for word in words:
            # Count words that could be keywords
            if len(word) > 3 and (word.isupper() or word.istitle() or self._is_domain_term(word)):
                potential_keywords += 1
        
        return potential_keywords / max(1, len(words))
    
    def _calculate_language_mixing(self, text: str) -> float:
        """Calculate how mixed the languages are in the text"""
        # Simple approach: look for alternating language patterns
        chinese_segments = 0
        english_segments = 0
        transitions = 0
        
        prev_was_chinese = None
        
        for char in text:
            is_chinese = '\u4e00' <= char <= '\u9fff'
            is_english = char.isalpha() and ord(char) < 128
            
            if is_chinese and prev_was_chinese is False:
                transitions += 1
                chinese_segments += 1
            elif is_english and prev_was_chinese is True:
                transitions += 1
                english_segments += 1
            
            if is_chinese or is_english:
                prev_was_chinese = is_chinese
        
        total_segments = chinese_segments + english_segments
        if total_segments == 0:
            return 0.0
        
        # More transitions = more mixing
        mixing_score = transitions / max(1, total_segments)
        return min(1.0, mixing_score)
    
    def _classify_content_domain(self, text: str) -> str:
        """Classify the domain of the content"""
        text_lower = text.lower()
        
        domain_keywords = {
            'technical': ['inspection', 'analysis', 'testing', 'specification', 'protocol', 'method'],
            'manufacturing': ['production', 'assembly', 'quality', 'defect', 'process', 'smt'],
            'engineering': ['design', 'development', 'system', 'engineering', 'validation'],
            'general': ['description', 'information', 'content', 'text', 'data']
        }
        
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            domain_scores[domain] = score
        
        # Return domain with highest score
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return 'general'
