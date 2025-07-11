from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from typing import List, Dict, Any
from models import Ticket, Entity
from extractors import EntityExtractor, SentimentAnalyzer, UrgencyClassifier


class TriageAgent:
    """Smart ticket triage agent that combines ML models with LLM reasoning."""
    
    def __init__(self, llm_model: str = "gpt-4o-mini"):
        # Initialize ML components
        self.entity_extractor = EntityExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.urgency_classifier = UrgencyClassifier()
        
        # Initialize PydanticAI agent
        self.agent = Agent(
            model=OpenAIModel(llm_model),
            result_type=Ticket,
            system_prompt=self._get_system_prompt()
        )
    
    def _get_system_prompt(self) -> str:
        return """
        You are a customer support triage assistant. Your job is to analyze customer emails 
        and create structured ticket information that can be used for automatic routing.
        
        You will be provided with:
        1. The original email text
        2. Pre-extracted entities from the text
        3. Sentiment analysis results
        4. Urgency classification
        
        Your task is to synthesize this information into a complete ticket structure.
        
        Guidelines:
        - Be concise but accurate in your summary
        - Choose the most appropriate next action based on the content
        - If no clear customer ID is found, generate a reasonable one based on email context
        - If no specific product is mentioned, infer from context or use "General Support"
        - Common next actions: "escalate_to_tier_2", "assign_to_billing", "technical_support", 
          "send_documentation", "schedule_call", "close_resolved"
        """
    
    async def process_email(self, email_text: str) -> Ticket:
        """Process a customer email and return a structured ticket."""
        
        # Extract information using ML models
        entities = self.entity_extractor.extract_entities(email_text)
        product_mentions = self.entity_extractor.extract_product_mentions(email_text)
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(email_text)
        urgency_result = self.urgency_classifier.classify_urgency(
            email_text, sentiment_result["sentiment"]
        )
        
        # Prepare context for the LLM
        context = {
            "email_text": email_text,
            "entities": [entity.dict() for entity in entities],
            "product_mentions": product_mentions,
            "sentiment": sentiment_result["sentiment"],
            "sentiment_confidence": sentiment_result["confidence"],
            "urgency": urgency_result["urgency"],
            "urgency_confidence": urgency_result["confidence"]
        }
        
        # Generate ticket using PydanticAI
        try:
            result = await self.agent.run(
                f"""
                Analyze this customer email and create a structured ticket:
                
                Email: {email_text}
                
                Pre-analysis results:
                - Entities found: {entities}
                - Product mentions: {product_mentions}
                - Sentiment: {sentiment_result['sentiment']} (confidence: {sentiment_result['confidence']:.2f})
                - Urgency: {urgency_result['urgency']} (confidence: {urgency_result['confidence']:.2f})
                
                Create a complete ticket with all required fields.
                """
            )
            
            # Update confidence score based on ML model confidence
            ticket_data = result.data.dict()
            ticket_data["confidence_score"] = (
                sentiment_result["confidence"] + urgency_result["confidence"]
            ) / 2
            
            return Ticket(**ticket_data)
            
        except Exception as e:
            # Fallback: create ticket with ML-only information
            return self._create_fallback_ticket(
                email_text, entities, product_mentions, 
                sentiment_result, urgency_result
            )
    
    def _create_fallback_ticket(
        self, 
        email_text: str, 
        entities: List[Entity],
        product_mentions: List[str],
        sentiment_result: Dict[str, Any],
        urgency_result: Dict[str, Any]
    ) -> Ticket:
        """Create a fallback ticket when LLM fails."""
        
        # Extract customer ID from entities or generate one
        customer_id = "UNKNOWN"
        for entity in entities:
            if entity.label in ["PERSON", "ORG"]:
                customer_id = f"C_{entity.text.replace(' ', '_').upper()}"
                break
        
        # Determine product
        product = product_mentions[0] if product_mentions else "General Support"
        
        # Create summary
        summary = email_text[:100] + "..." if len(email_text) > 100 else email_text
        
        # Determine next action
        next_action = "escalate_to_tier_2" if urgency_result["urgency"] == "high" else "assign_to_support"
        
        return Ticket(
            customer_id=customer_id,
            product=product,
            sentiment=sentiment_result["sentiment"],
            urgency=urgency_result["urgency"],
            entities=entities,
            summary=summary,
            next_action=next_action,
            confidence_score=(sentiment_result["confidence"] + urgency_result["confidence"]) / 2
        )
    
    def process_email_sync(self, email_text: str) -> Ticket:
        """Synchronous wrapper for process_email."""
        import asyncio
        return asyncio.run(self.process_email(email_text))