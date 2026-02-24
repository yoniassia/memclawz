#!/usr/bin/env python3
"""
QMDZvec Context Optimizer
Analyzes an OpenClaw workspace and recommends how to split into
sub-agents with shared memory but minimal individual context.

Usage: python3 scripts/optimize-context.py [workspace_path]

Output: Creates <workspace>/memory/context-optimization.md with:
  1. Current context analysis (what's bloating your prompt)
  2. Recommended sub-agent architecture
  3. Skill distribution across agents
  4. Slim AGENTS.md draft for main orchestrator
  5. Ready-to-use sub-agent config templates
"""

import os
import sys
import json
import glob
from collections import defaultdict
from pathlib import Path

WORKSPACE = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.openclaw/workspace")

# Token estimate: ~1 token per 4 chars
def tokens(text):
    return len(text) // 4

def file_tokens(path):
    try:
        return tokens(open(path).read())
    except:
        return 0

def categorize_skill(name):
    """Categorize a skill by its name into a domain."""
    name_l = name.lower()
    
    # Trading/Finance
    if any(w in name_l for w in ["trade", "swap", "defi", "binance", "etoro", "finance",
        "stock", "forex", "crypto", "wallet", "solana", "evm", "uniswap", "hyperliquid",
        "perpetual", "polymarket", "prediction", "portfolio", "quant", "bloomberg",
        "yahoo", "tushare", "fred", "factset", "gurufocus", "allium", "einstein",
        "aave", "0x", "openocean", "dex", "blockchain"]):
        return "finance"
    
    # Marketing/Content
    if any(w in name_l for w in ["market", "seo", "copy", "content", "brand", "campaign",
        "email", "social", "ad", "cro", "popup", "pricing", "launch", "referral",
        "competitor", "ab-test", "analytics", "paid", "programmatic", "schema",
        "signup", "form", "paywall", "onboarding", "etoro-brand", "etoro-compliance"]):
        return "marketing"
    
    # Development/Code
    if any(w in name_l for w in ["code", "dev", "git", "ci-cd", "docker", "test",
        "review", "clean", "codebase", "e2e", "playwright", "lint", "qa",
        "pr-reviewer", "senior-qa", "test-master", "preflight"]):
        return "development"
    
    # Infrastructure/DevOps
    if any(w in name_l for w in ["cloud", "hetzner", "devops", "dns", "domain",
        "server", "nginx", "deploy", "infra"]):
        return "infrastructure"
    
    # Communication/Outreach
    if any(w in name_l for w in ["email", "telegram", "whatsapp", "wacli", "phone",
        "call", "voice", "tts", "calendar", "agentmail", "x-twitter"]):
        return "communication"
    
    # Research/Analysis
    if any(w in name_l for w in ["research", "browse", "browser", "web", "scrape",
        "analyze", "survey", "interview", "feedback", "evals", "ai-"]):
        return "research"
    
    # Product/Strategy
    if any(w in name_l for w in ["product", "strategy", "vision", "roadmap", "team",
        "culture", "career", "coach", "delegat", "energy", "promot", "sales",
        "enterprise", "community", "design-system", "behavioral"]):
        return "strategy"
    
    return "general"

def analyze():
    print(f"🔍 Analyzing workspace: {WORKSPACE}\n")
    
    # === 1. Context File Analysis ===
    context_files = {}
    for name in ["AGENTS.md", "SOUL.md", "USER.md", "MEMORY.md", "TOOLS.md",
                  "IDENTITY.md", "HEARTBEAT.md"]:
        path = os.path.join(WORKSPACE, name)
        if os.path.exists(path):
            content = open(path).read()
            context_files[name] = {
                "bytes": len(content),
                "tokens": tokens(content),
                "lines": content.count("\n"),
            }
    
    total_tokens = sum(v["tokens"] for v in context_files.values())
    
    # === 2. Skill Analysis ===
    skills = defaultdict(list)
    skill_dirs = []
    
    for skills_dir in [os.path.join(WORKSPACE, "skills"),
                       os.path.expanduser("~/.npm-global/lib/node_modules/openclaw/skills")]:
        if os.path.isdir(skills_dir):
            for d in sorted(os.listdir(skills_dir)):
                skill_path = os.path.join(skills_dir, d, "SKILL.md")
                if os.path.exists(skill_path):
                    category = categorize_skill(d)
                    skills[category].append(d)
                    skill_dirs.append(d)
    
    # === 3. Memory Analysis ===
    memory_files = glob.glob(os.path.join(WORKSPACE, "memory/*.md"))
    knowledge_files = glob.glob(os.path.join(WORKSPACE, "knowledge/**/*.md"), recursive=True)
    qmd_path = os.path.join(WORKSPACE, "memory/qmd/current.json")
    qmd_exists = os.path.exists(qmd_path)
    
    # === 4. Generate Report ===
    report = []
    report.append("# QMDZvec Context Optimization Report\n")
    report.append(f"*Generated for workspace: `{WORKSPACE}`*\n")
    report.append("---\n")
    
    # Current state
    report.append("## 1. Current Context Load\n")
    report.append("| File | Bytes | Tokens | % of Total |")
    report.append("|------|-------|--------|------------|")
    for name, info in sorted(context_files.items(), key=lambda x: -x[1]["tokens"]):
        pct = info["tokens"] / total_tokens * 100 if total_tokens else 0
        report.append(f"| {name} | {info['bytes']:,} | {info['tokens']:,} | {pct:.0f}% |")
    report.append(f"| **Total** | **{sum(v['bytes'] for v in context_files.values()):,}** | **{total_tokens:,}** | **100%** |")
    report.append("")
    
    # Bloat analysis
    report.append("### Bloat Sources\n")
    if context_files.get("AGENTS.md", {}).get("tokens", 0) > 2000:
        report.append(f"- ⚠️ **AGENTS.md is {context_files['AGENTS.md']['tokens']:,} tokens** — should be <1,500 for an orchestrator")
    if context_files.get("MEMORY.md", {}).get("tokens", 0) > 1500:
        report.append(f"- ⚠️ **MEMORY.md is {context_files['MEMORY.md']['tokens']:,} tokens** — move details to Zvec, keep only essentials")
    if context_files.get("USER.md", {}).get("tokens", 0) > 500:
        report.append(f"- ⚠️ **USER.md is {context_files['USER.md']['tokens']:,} tokens** — contact details should be in Zvec, not prompt")
    report.append(f"- 📊 Skill descriptions in prompt: ~{len(skill_dirs) * 40} tokens (est. {min(70, len(skill_dirs))} of {len(skill_dirs)} loaded)")
    report.append("")
    
    # Skill distribution
    report.append("## 2. Skill Distribution\n")
    report.append(f"**Total skills: {len(skill_dirs)}**\n")
    report.append("| Domain | Count | Skills |")
    report.append("|--------|-------|--------|")
    for cat in sorted(skills.keys(), key=lambda c: -len(skills[c])):
        skill_list = ", ".join(skills[cat][:8])
        if len(skills[cat]) > 8:
            skill_list += f", +{len(skills[cat])-8} more"
        report.append(f"| {cat.title()} | {len(skills[cat])} | {skill_list} |")
    report.append("")
    
    # Recommended architecture
    report.append("## 3. Recommended Sub-Agent Architecture\n")
    report.append("```")
    report.append("🦞 Main Orchestrator (~1,500 tokens)")
    report.append("│   Identity + routing + QMDZvec protocol")
    report.append("│   NO skill descriptions, NO detailed contacts")
    report.append("│")
    
    agent_configs = []
    
    for cat, cat_skills in sorted(skills.items(), key=lambda x: -len(x[1])):
        if cat == "general" and len(cat_skills) < 5:
            continue
        emoji = {"finance": "💰", "marketing": "🎯", "development": "🔨",
                "infrastructure": "🏗️", "communication": "📬", "research": "🔍",
                "strategy": "📋", "general": "🔧"}.get(cat, "🔧")
        name = {"finance": "TradeClaw", "marketing": "MarketClaw",
                "development": "DevClaw", "infrastructure": "InfraClaw",
                "communication": "CommsClaw", "research": "ResearchClaw",
                "strategy": "StrategyClaw", "general": "UtilityClaw"}.get(cat, f"{cat.title()}Claw")
        
        report.append(f"├── {emoji} {name} ({len(cat_skills)} skills, ~2K tokens)")
        agent_configs.append({"name": name, "emoji": emoji, "category": cat,
                            "skills": cat_skills, "token_budget": 2000})
    
    report.append("│")
    report.append("└── 🧠 Shared QMDZvec Fleet Memory (all agents)")
    report.append("        QMD + Zvec search + auto-sync")
    report.append("```\n")
    
    # Token budget
    report.append("## 4. Token Budget\n")
    report.append("| Component | Current | Proposed | Savings |")
    report.append("|-----------|---------|----------|---------|")
    
    agents_current = context_files.get("AGENTS.md", {}).get("tokens", 0)
    memory_current = context_files.get("MEMORY.md", {}).get("tokens", 0)
    user_current = context_files.get("USER.md", {}).get("tokens", 0)
    soul_current = context_files.get("SOUL.md", {}).get("tokens", 0)
    skill_current = min(70, len(skill_dirs)) * 40
    
    report.append(f"| AGENTS.md | {agents_current:,} | 1,200 | -{agents_current - 1200:,} |")
    report.append(f"| MEMORY.md | {memory_current:,} | 800 | -{memory_current - 800:,} |")
    report.append(f"| USER.md | {user_current:,} | 300 | -{user_current - 300:,} |")
    report.append(f"| SOUL.md | {soul_current:,} | {min(soul_current, 500)} | -{max(0, soul_current - 500):,} |")
    report.append(f"| Skill descriptions | {skill_current:,} | 0 | -{skill_current:,} |")
    report.append(f"| System overhead | ~3,500 | ~3,500 | 0 |")
    proposed = 1200 + 800 + 300 + min(soul_current, 500) + 3500
    report.append(f"| **Total** | **{total_tokens + skill_current + 3500:,}** | **{proposed:,}** | **-{total_tokens + skill_current + 3500 - proposed:,}** |")
    report.append("")
    
    # Slim AGENTS.md draft
    report.append("## 5. Slim AGENTS.md Draft (for Main Orchestrator)\n")
    report.append("```markdown")
    report.append("# AGENTS.md — Orchestrator")
    report.append("")
    report.append("## Memory Protocol (QMDZvec)")
    report.append("1. On session start: Read `memory/qmd/current.json`")
    report.append("2. During work: Update QMD after significant actions")
    report.append("3. For recall: QMD (<1ms) → Zvec localhost:4010 (<10ms) → memory_search")
    report.append("4. On session end: Run `python3 QMDZvec/scripts/qmd-compact.py`")
    report.append("")
    report.append("## Sub-Agent Routing")
    report.append("Don't do everything yourself. Spawn specialists:")
    for ac in agent_configs:
        triggers = ", ".join(ac["skills"][:5])
        report.append(f"- **{ac['emoji']} {ac['name']}** — {ac['category']}: {triggers}")
    report.append("")
    report.append("## Rules")
    report.append("- Read before edit. Test before deploy. Ask before delete.")
    report.append("- Security issues → Haim. External comms → draft first.")
    report.append("- Write to memory files DURING work, not after.")
    report.append("```\n")
    
    # Sub-agent templates
    report.append("## 6. Sub-Agent Config Templates\n")
    for ac in agent_configs:
        report.append(f"### {ac['emoji']} {ac['name']}\n")
        report.append("```markdown")
        report.append(f"# AGENTS.md — {ac['name']}")
        report.append(f"Specialist in: {ac['category']}")
        report.append(f"Skills: {', '.join(ac['skills'][:15])}")
        if len(ac['skills']) > 15:
            report.append(f"  + {len(ac['skills'])-15} more")
        report.append("")
        report.append("## Memory")
        report.append("- Read QMD on start: `memory/qmd/current.json`")
        report.append("- Search Zvec: POST http://localhost:4010/search")
        report.append("- Write results to QMD when done")
        report.append("```\n")
    
    # Migration steps
    report.append("## 7. Migration Plan\n")
    report.append("### Phase 1: Slim Context (30 min)")
    report.append("1. Replace AGENTS.md with slim version (Section 5 above)")
    report.append("2. Move detailed contacts from USER.md → `knowledge/people/contacts.md`")
    report.append("3. Move project details from MEMORY.md → QMD or Zvec-searchable files")
    report.append("4. Remove skill descriptions from system prompt (use sub-agent routing)\n")
    report.append("### Phase 2: Create Sub-Agents (1 hour)")
    report.append("1. Create workspace dirs for each sub-agent")
    report.append("2. Copy relevant skills to each")
    report.append("3. Point all to shared QMDZvec fleet server")
    report.append("4. Test spawning from main → sub-agent\n")
    report.append("### Phase 3: Shared Memory (30 min)")
    report.append("1. Start QMDZvec fleet server with namespaces")
    report.append("2. Configure each sub-agent's Zvec endpoint")
    report.append("3. Test cross-agent memory search\n")
    
    # Write report
    output_path = os.path.join(WORKSPACE, "memory", "context-optimization.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(report))
    
    # Print summary
    print("📊 CONTEXT ANALYSIS")
    print("=" * 50)
    print(f"  Context files: {total_tokens:,} tokens")
    print(f"  Skills: {len(skill_dirs)} ({len(skills)} domains)")
    print(f"  Memory files: {len(memory_files)} daily logs")
    print(f"  Knowledge files: {len(knowledge_files)}")
    print(f"  QMD: {'✅' if qmd_exists else '❌ not set up'}")
    print()
    print("💡 RECOMMENDATIONS")
    print("=" * 50)
    print(f"  Current prompt: ~{total_tokens + skill_current + 3500:,} tokens")
    print(f"  Optimized:      ~{proposed:,} tokens")
    print(f"  Savings:         {(1 - proposed/(total_tokens + skill_current + 3500))*100:.0f}%")
    print(f"  Sub-agents:      {len(agent_configs)}")
    for ac in agent_configs:
        print(f"    {ac['emoji']} {ac['name']}: {len(ac['skills'])} skills")
    print()
    print(f"📝 Full report: {output_path}")
    print(f"\n✅ Run this to apply: review the report, then follow Phase 1-3")

if __name__ == "__main__":
    analyze()
