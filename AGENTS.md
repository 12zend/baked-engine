# AGENTS.md

## Purpose
This file defines how coding agents should work in this repository. Implement changes incrementally and **create Git commits at meaningful, self-contained milestones**, rather than waiting until the entire task is complete.

## Core workflow
1. Before editing, read the relevant documentation and inspect the repository, current branch, `git status`, and recent commits.
2. Break the requested work into small, independently verifiable milestones. Prefer milestones that leave the repository in a working state.
3. Implement one milestone at a time. Keep changes focused and avoid unrelated refactoring.
4. At each milestone, run the relevant checks (format, lint, typecheck, tests, build, or a targeted manual check). Fix failures caused by the changes before committing.
5. Review `git diff` and `git status`. Stage **only files related to that milestone**, then create a commit with a descriptive message.
6. Continue to the next milestone and repeat. At the end, report the changes, checks performed, commit hashes, and any remaining issues.

## When to commit
Commit at a natural boundary, such as:
- Completing a coherent feature or subfeature.
- Finishing a bug fix with regression coverage.
- Adding or updating a discrete module, API, or UI component.
- Completing a migration or configuration change that can be verified independently.
- Finishing a documentation update associated with a completed change.

**Do not** commit after every file edit or arbitrary number of lines. **Do not** postpone all commits until the end of a multi-stage task. For long tasks, seek several meaningful checkpoints.

## Commit quality rules
- Each commit should represent one understandable logical change.
- Prefer a clean, testable state at every commit; do not knowingly commit broken code unless the task explicitly requires an intermediate nonworking state and that state is clearly identified.
- Use concise commit messages in the format `type(scope): summary` where useful. Typical types: `feat`, `fix`, `refactor`, `test`, `docs`, `build`, `chore`.
- Keep commits easy to review and revert independently.
- Never include secrets, credentials, local environment files, build artifacts, or unrelated changes.
- Do not amend, squash, rebase, reset, or overwrite existing commits unless explicitly instructed.
- Never push to a remote or create a release unless explicitly requested.

## Handling existing changes
- Treat pre-existing uncommitted changes as user-owned. Do not discard, rewrite, or accidentally commit them.
- If the working tree is dirty, isolate and stage only the agent's own modifications (use selective staging when needed).
- Avoid destructive Git commands such as `git reset --hard`, `git clean -fd`, and force-push.
- If safe separation of changes is impossible, explain the blocker instead of creating a mixed commit.

## Verification and exceptions
- Run the most relevant available verification before each commit; do not claim checks passed unless they actually ran successfully.
- If a check cannot run due to missing tooling or external dependencies, document the limitation and run whatever local checks are feasible.
- If a milestone cannot be safely committed, leave its changes uncommitted, explain why, and continue only when doing so is safe.
- Follow any repository-specific instructions and user requirements that are more specific than this file.

## Final report
Provide a concise summary including:
- Completed milestones and corresponding commit hashes/messages.
- Tests/builds/checks executed and their results.
- Uncommitted changes (if any), blockers, and recommended next steps.
