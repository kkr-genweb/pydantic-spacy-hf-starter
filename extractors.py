import spacy
from transformers import pipeline
from typing import List, Dict, Any
from models import Entity
import os
import warnings

# Suppress PyTorch MPS warnings and device messages
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
warnings.filterwarnings("ignore", message=".*MPS.*")

# Suppress transformers warnings
warnings.filterwarnings("ignore", message=".*return_all_scores.*")

# Redirect stdout/stderr to suppress device messages (optional)
import sys
from contextlib import redirect_stdout, redirect_stderr
import io

# Optional: Force CPU usage instead of MPS (uncomment if needed)
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# import torch
# torch.set_default_device('cpu')


class EntityExtractor:
    """Extracts named entities using spaCy transformer model."""
    
    def __init__(self, model_name: str = "en_core_web_trf"):
        self.nlp = spacy.load(model_name)
        
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text."""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=1.0  # spaCy doesn't provide confidence scores by default
            )
            entities.append(entity)
            
        return entities
    
    def extract_product_mentions(self, text: str) -> List[str]:
        """Extract potential product mentions from text."""
        doc = self.nlp(text)
        products = []
        
        # Look for product-like entities
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG", "PERSON"]:
                products.append(ent.text)
                
        # Also look for common product patterns
        for token in doc:
            if token.pos_ == "NOUN" and token.text.lower() in ["widget", "app", "software", "service"]:
                products.append(token.text)
                
        return list(set(products))  # Remove duplicates


class SentimentAnalyzer:
    """Analyzes sentiment using Hugging Face transformers."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            return_all_scores=True
        )
        
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        results = self.sentiment_pipeline(text)[0]
        
        # Convert to our format
        sentiment_map = {"POSITIVE": "positive", "NEGATIVE": "negative"}
        
        max_score = max(results, key=lambda x: x["score"])
        sentiment = sentiment_map.get(max_score["label"], "neutral")
        
        return {
            "sentiment": sentiment,
            "confidence": max_score["score"],
            "scores": {result["label"]: result["score"] for result in results}
        }


class UrgencyClassifier:
    """Classifies urgency based on text patterns and keywords."""
    
    def __init__(self):
        self.high_urgency_keywords = [
            "urgent", "emergency", "asap", "immediately", "critical",
            "broken", "down", "not working", "error", "bug", "crash",
            "angry", "frustrated", "disappointed", "unacceptable"
        ]
        
        self.medium_urgency_keywords = [
            "soon", "issue", "problem", "question", "help",
            "support", "assistance", "confused", "unclear"
        ]
        
    def classify_urgency(self, text: str, sentiment: str) -> Dict[str, Any]:
        """Classify urgency based on text content and sentiment."""
        text_lower = text.lower()
        
        high_score = sum(1 for keyword in self.high_urgency_keywords if keyword in text_lower)
        medium_score = sum(1 for keyword in self.medium_urgency_keywords if keyword in text_lower)
        
        # Adjust score based on sentiment
        if sentiment == "negative":
            high_score += 2
        elif sentiment == "positive":
            high_score -= 1
            
        # Determine urgency
        if high_score >= 3:
            urgency = "high"
            confidence = min(0.9, 0.5 + (high_score * 0.1))
        elif high_score >= 1 or medium_score >= 2:
            urgency = "medium"
            confidence = min(0.8, 0.4 + (high_score + medium_score) * 0.1)
        else:
            urgency = "low"
            confidence = 0.6
            
        return {
            "urgency": urgency,
            "confidence": confidence,
            "keyword_matches": {
                "high": high_score,
                "medium": medium_score
            }
        }