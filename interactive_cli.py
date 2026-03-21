#!/usr/bin/env python
"""
Interactive SkillBridge CLI
Walks you through the full user journey using your local server.
"""

import os
import sys
from pathlib import Path
import httpx
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
BASE_URL = "http://localhost:8000"

def check_server():
    try:
        httpx.get(f"{BASE_URL}/health")
        return True
    except:
        return False

def get_file_content(prompt_text):
    while True:
        path_str = Prompt.ask(prompt_text)
        path = Path(path_str)
        if path.exists() and path.is_file():
            return path.read_text(encoding="utf-8")
        console.print(f"[red]Error: File not found at {path_str}[/red]")

def main():
    console.print(Panel.fit("🛠️  [bold blue]SkillBridge Interactive Testing Tool[/bold blue]"))

    if not check_server():
        console.print("[red]Error: Local server not detected at http://localhost:8000[/red]")
        console.print("[yellow]Please run 'uv run python -m app.main' in another terminal first.[/yellow]")
        sys.exit(1)

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        
        # --- PHASE 1: HR SETUP ---
        console.print("\n[bold green]--- Phase 1: HR Job Description Setup ---[/bold green]")
        role_title = Prompt.ask("Enter Job Title", default="Software Engineer")
        company = Prompt.ask("Enter Company Name", default="My Company")
        domain = Prompt.ask("Enter Domain", choices=["technical", "operations"], default="technical")
        jd_text = get_file_content("Enter path to JD .txt file")

        with console.status("[bold green]Extracting skills from JD..."):
            resp = client.post("/api/jd/upload", json={
                "raw_text": jd_text,
                "role_title": role_title,
                "company": company,
                "domain": domain
            })
            draft_skills = resp.json()["draft_skills"]

        console.print(f"✓ Extracted {len(draft_skills)} skills.")
        if Confirm.ask("Do you want to confirm these skills and save the JD?"):
            resp = client.post("/api/jd/confirm", json={
                "role_title": role_title,
                "company": company,
                "domain": domain,
                "raw_text": jd_text,
                "required_skills": draft_skills
            })
            jd_id = resp.json()["jd_id"]
            console.print(f"[bold green]✓ JD Saved! ID: {jd_id}[/bold green]")
        else:
            console.print("[red]Aborted.[/red]")
            return

        # --- PHASE 2: CANDIDATE RESUME ---
        console.print("\n[bold magenta]--- Phase 2: Candidate Resume Processing ---[/bold magenta]")
        resume_text = get_file_content("Enter path to your Resume .txt file")

        with console.status("[bold magenta]Processing Resume & Computing Mastery (Groq)..."):
            resp = client.post("/api/resume/upload", json={"raw_text": resume_text})
            extracted_skills = resp.json()["extracted_skills"]

        table = Table(title="Extracted Skills & Mastery", show_header=True, header_style="bold cyan")
        table.add_column("Skill")
        table.add_column("Mastery", justify="right")
        for s in sorted(extracted_skills, key=lambda x: x['mastery_score'], reverse=True)[:10]:
            table.add_row(s['label'], f"{s['mastery_score']:.2f}")
        console.print(table)

        if Confirm.ask("Do you confirm these skills?"):
            resp = client.post("/api/resume/confirm", json={
                "raw_text": resume_text,
                "confirmed_skills": extracted_skills
            })
            session_id = resp.json()["current_state_id"]
            console.print(f"[bold green]✓ Session Created! ID: {session_id}[/bold green]")
        else:
            console.print("[red]Aborted.[/red]")
            return

        # --- PHASE 3: DYNAMIC QUESTIONS ---
        console.print("\n[bold yellow]--- Phase 3: Preference Survey ---[/bold yellow]")
        resp = client.post("/api/pathway/questions", json={
            "current_state_id": session_id,
            "jd_id": jd_id
        })
        questions = resp.json()["questions"]
        
        console.print(f"System generated {len(questions)} preferences questions for you:")
        answers = {}
        for q in questions:
            console.print(f"\n[bold]{q['text']}[/bold]")
            choice = Prompt.ask("Choose one", choices=q['options'])
            answers[q['id']] = choice

        # --- PHASE 4: FINAL PATHWAY ---
        console.print("\n[bold blue]--- Phase 4: Final Pathway Generation ---[/bold blue]")
        with console.status("[bold blue]Running Priority Pathing Algorithm..."):
            resp = client.post("/api/pathway/generate", json={
                "current_state_id": session_id,
                "jd_id": jd_id,
                "preferences": answers
            })
        
        if resp.status_code != 200:
            console.print(f"[red]Error: {resp.text}[/red]")
            return

        result = resp.json()
        
        # Display Final Report
        console.print(Panel.fit("✅ [bold green]LEARNING PATHWAY READY[/bold green]"))
        
        pathway_table = Table(show_header=True, header_style="bold blue")
        pathway_table.add_column("Step")
        pathway_table.add_column("Course ID")
        pathway_table.add_column("Status")
        pathway_table.add_column("Mastery Signal", justify="right")

        for i, course in enumerate(result["final_pathway"], 1):
            st = course["node_state"].upper()
            color = "green" if st == "SKIPPED" else "yellow" if st == "ASSIGNED" else "blue"
            pathway_table.add_row(
                str(i),
                course["course_id"],
                f"[{color}]{st}[/]",
                f"{course['mastery_score']:.2f}"
            )
        
        console.print(pathway_table)
        console.print(f"\n[bold cyan]Efficiency Boost:[/bold cyan] [bold green]{result['metrics']['reduction_pct']}%[/bold green] of unnecessary content removed.")
        console.print("\n[bold blue]Test complete![/bold blue] ✨")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]Session ended.[/red]")
