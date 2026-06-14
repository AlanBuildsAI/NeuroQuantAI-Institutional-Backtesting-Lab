# Codex / Claude Code Next Steps

Use this prompt if you want to keep improving the repository from VS Code with Codex or Claude Code.

```text
You are helping improve a portfolio analytics repository called NeuroQuantAI Institutional Backtesting Lab.

Goal: make it look like a polished data analytics case study, not a trading product.

Constraints:
- Use synthetic data only.
- Do not add live data integrations.
- Do not make prediction or performance claims.
- Keep the project safe, reproducible, and recruiter-friendly.

Tasks:
1. Add a notebook-style walkthrough that explains each step of the analytics workflow.
2. Add simple unit tests for data validation, parameter validation, and output columns.
3. Add an HTML report generator that reads sample_outputs/parameter_sweep_summary.csv and creates a clean one-page report.
4. Add a Makefile or simple task runner with commands: install, run, test, clean.
5. Keep README concise and aligned to data analyst / operations analyst roles.
6. Do not over-engineer the repo.

Output expected:
- updated code
- updated README if needed
- tests
- short summary of changes
```
