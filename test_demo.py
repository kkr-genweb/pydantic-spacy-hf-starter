#!/usr/bin/env python3
"""
Test script to demonstrate the ticket triage system without requiring OpenAI API
"""

import json
from extractors import EntityExtractor, SentimentAnalyzer, UrgencyClassifier
from models import Ticket, Entity
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.json import JSON


def create_demo_ticket(email_text: str) -> Ticket:
    """Create a demo ticket using only the ML components (no LLM)."""
    
    # Initialize extractors
    entity_extractor = EntityExtractor()
    sentiment_analyzer = SentimentAnalyzer()
    urgency_classifier = UrgencyClassifier()
    
    # Extract information
    entities = entity_extractor.extract_entities(email_text)
    product_mentions = entity_extractor.extract_product_mentions(email_text)
    sentiment_result = sentiment_analyzer.analyze_sentiment(email_text)
    urgency_result = urgency_classifier.classify_urgency(
        email_text, sentiment_result["sentiment"]
    )
    
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
    
    # Determine next action based on urgency and sentiment
    if urgency_result["urgency"] == "high":
        next_action = "escalate_to_tier_2"
    elif sentiment_result["sentiment"] == "negative":
        next_action = "assign_to_senior_support"
    elif "billing" in email_text.lower() or "payment" in email_text.lower():
        next_action = "assign_to_billing"
    else:
        next_action = "assign_to_support"
    
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


def display_ticket(console: Console, ticket: Ticket, email_text: str):
    """Display ticket information in a formatted way."""
    
    # Original email panel
    console.print(Panel(
        email_text[:300] + "..." if len(email_text) > 300 else email_text,
        title="📧 Original Email",
        border_style="blue"
    ))
    
    # Ticket information table
    table = Table(title="🎫 Ticket Analysis", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", width=15)
    table.add_column("Value", style="white")
    
    table.add_row("Customer ID", ticket.customer_id)
    table.add_row("Product", ticket.product)
    table.add_row("Sentiment", get_sentiment_display(ticket.sentiment))
    table.add_row("Urgency", get_urgency_display(ticket.urgency))
    table.add_row("Summary", ticket.summary)
    table.add_row("Next Action", ticket.next_action)
    table.add_row("Confidence", f"{ticket.confidence_score:.2%}")
    
    console.print(table)
    
    # Entities panel
    if ticket.entities:
        entities_text = "\n".join([
            f"• {entity.text} ({entity.label})"
            for entity in ticket.entities
        ])
        console.print(Panel(
            entities_text,
            title="🏷️ Extracted Entities",
            border_style="green"
        ))
    
    # JSON output panel
    console.print(Panel(
        JSON.from_data(ticket.dict()),
        title="📄 JSON Output",
        border_style="yellow"
    ))


def get_sentiment_display(sentiment: str) -> str:
    """Get colored sentiment display."""
    colors = {
        "positive": "😊 [green]Positive[/green]",
        "negative": "😠 [red]Negative[/red]",
        "neutral": "😐 [yellow]Neutral[/yellow]"
    }
    return colors.get(sentiment, sentiment)


def get_urgency_display(urgency: str) -> str:
    """Get colored urgency display."""
    colors = {
        "high": "🚨 [red]High[/red]",
        "medium": "⚠️ [yellow]Medium[/yellow]",
        "low": "✅ [green]Low[/green]"
    }
    return colors.get(urgency, urgency)


def main():
    console = Console()
    
    console.print("🤖 Smart Ticket Triage Agent Demo", style="bold blue")
    console.print("Using ML-only mode (no OpenAI API required)\n")
    
    # Load sample emails
    try:
        with open("sample_emails.txt", "r") as f:
            content = f.read()
    except FileNotFoundError:
        console.print("❌ sample_emails.txt not found", style="red")
        return
    
    # Split emails by separator
    emails = content.split("---")
    
    for i, email_text in enumerate(emails, 1):
        email_text = email_text.strip()
        if not email_text:
            continue
            
        console.print(f"\n{'='*60}")
        console.print(f"Processing Email {i}/{len(emails)}")
        console.print(f"{'='*60}")
        
        try:
            ticket = create_demo_ticket(email_text)
            display_ticket(console, ticket, email_text)
            
        except Exception as e:
            console.print(f"❌ Error processing email {i}: {e}", style="red")
        
        if i < len(emails) - 1:
            input("\nPress Enter to continue to next email...")


if __name__ == "__main__":
    main()