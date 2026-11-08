# Python Development Skill

## Purpose

Create clear, well-documented, maintainable code. This skill guides systematic development or refactoring that prioritizes human comprehension without sacrificing correctness or reasonable performance.

## When to Invoke

Invoke this skill when:
- User explicitly requests "human", "readable", "maintainable", "clean", or "refactor" code 
- Code review processes flag comprehension or maintainability issues
- Working with legacy code that needs modernization
- Preparing code for team onboarding or educational contexts
- Code complexity metrics exceed reasonable thresholds
- Functions or modules are difficult to understand or modify
- General code development where clarity and maintainability are priorities
- **🚨 RED FLAG: You see any of these spaghetti code indicators:**
  - File >500 lines with scattered functions and global state
  - Multiple `global` statements throughout the code
  - Functions with hard-coded dependencies (no dependency injection)
  - No clear module/class organization
  - Dict/list manipulation instead of proper domain objects
  - Configuration mixed with business logic

Do NOT invoke this skill when:
- Code is performance-critical and profiling shows optimization is needed first
- Code is scheduled for deletion or replacement
- External dependencies require upstream contributions instead
- User explicitly requests performance optimization over readability

## Core Principles

Follow these principles in priority order:

1. **CLASS-BASED ARCHITECTURE IS MANDATORY** - **ALWAYS prefer proper class-based OOP architecture over long messy spaghetti code scripts.** Transform any procedural "script-like" code into well-structured OOP with proper classes, modules, and interfaces. This is NON-NEGOTIABLE.
   - ❌ **NEVER accept**: Globals, scattered functions, 1000+ line scripts, no clear structure
   - ✅ **ALWAYS create**: Classes with clear responsibilities, dependency injection, proper modules
2. **Clarity over cleverness** - Explicit, obvious code beats implicit, clever code
3. **Preserve correctness** - All tests must pass; behavior must remain identical
4. **Single Responsibility** - Each class and function should do one thing well (SOLID principles)
5. **Self-documenting structure** - Code structure tells what, comments explain why
6. **Progressive disclosure** - Reveal complexity in layers, not all at once
7. **Reasonable performance** - Never sacrifice >2x performance without explicit approval

## Key Constraints

ALWAYS observe these constraints:
- **OOP-ORIENTED ARCHITECTURE** - Transform script-like code into proper OOP with classes, modules, and clear boundaries. Avoid global state, scattered functions, and spaghetti code
- **PRESERVE BEHAVIOR** - All existing tests must pass after refactoring
- **NO PERFORMANCE REGRESSION** - Never degrade performance >2x without explicit user approval
- **NO API CHANGES** - Public APIs remain unchanged unless explicitly requested and documented
- **NO OVER-ENGINEERING** - Simple code stays simple; don't add unnecessary abstraction (but DO structure code properly)
- **NO MAGIC** - No framework magic, no metaprogramming unless absolutely necessary
