#!/usr/bin/env python3
"""
Smart Ticket Triage Agent CLI
Usage: python triage.py [email_file] or echo "email text" | python triage.py
"""

import sys
import json
import argparse
from pathlib import Path
from agent import TriageAgent
from test_demo import create_demo_ticket
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
from rich.table import Table


def main():
    console = Console()
    
    parser = argparse.ArgumentParser(description="Smart Ticket Triage Agent")
    parser.add_argument("file", nargs="?", help="Email file to process")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use")
    parser.add_argument("--demo", action="store_true", help="Use demo mode (no API key required)")
    args = parser.parse_args()
    
    # Initialize the agent (only if not using demo mode)
    if not args.demo:
        try:
            console.print("🤖 Initializing Ticket Triage Agent...", style="blue")
            agent = TriageAgent(llm_model=args.model)
            console.print("✅ Agent initialized successfully!", style="green")
        except Exception as e:
            console.print(f"❌ Failed to initialize agent: {e}", style="red")
            console.print("💡 Try using --demo mode to run without API keys", style="yellow")
            sys.exit(1)
    else:
        console.print("🤖 Running in demo mode (no API key required)", style="blue")
        agent = None
    
    # Get email text
    if args.file:
        try:
            email_text = Path(args.file).read_text()
        except Exception as e:
            console.print(f"❌ Error reading file: {e}", style="red")
            sys.exit(1)
    else:
        # Read from stdin
        email_text = sys.stdin.read().strip()
        if not email_text:
            console.print("❌ No email text provided", style="red")
            sys.exit(1)
    
    # Process the email
    try:
        console.print("📧 Processing email...", style="blue")
        
        if args.demo:
            # Use demo mode (no API key required)
            ticket = create_demo_ticket(email_text)
        else:
            # Use full LLM mode
            ticket = agent.process_email_sync(email_text)
        
        if args.json:
            # Output as JSON
            print(json.dumps(ticket.model_dump(), indent=2))
        else:
            # Rich formatted output
            display_ticket(console, ticket, email_text)
            
    except Exception as e:
        console.print(f"❌ Error processing email: {e}", style="red")
        sys.exit(1)


def display_ticket(console: Console, ticket, email_text: str):
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
        JSON.from_data(ticket.model_dump()),
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


if __name__ == "__main__":
    main()