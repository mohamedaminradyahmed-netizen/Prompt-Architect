# Prompt Architect - Project Structure

## Directory Organization

### Root Level
```
e:\Prompt-Architect/
├── src/                    # Core source code modules
├── .amazonq/              # Amazon Q configuration and rules
├── .claude/               # Claude AI settings
├── *.ts                   # Main application files
├── *.tsx                  # React components
├── *.md                   # Documentation and planning
└── package.json           # Project dependencies and scripts
```

### Source Code Structure (`src/`)
```
src/
├── __tests__/             # Unit tests and test utilities
├── config/                # Configuration modules
│   ├── balanceMetrics.ts     # Balance metrics definitions
│   ├── balanceMetrics.example.ts  # Example configurations
│   └── README.md             # Configuration documentation
├── templates/             # Prompt template system
│   ├── PromptTemplate.ts     # Template interface and types
│   └── templateParser.ts     # Template parsing utilities
└── mutations.ts           # Mutation operators implementation
```

### Main Application Files
- `mutations.ts` - Core mutation operators (3 basic types)
- `evaluator.ts` - Evaluation system with heuristic scoring
- `prompt-engineer.tsx` - React UI component for prompt optimization

## Core Components & Relationships

### 1. Mutation System
**Location**: `src/mutations.ts`, `mutations.ts`
**Purpose**: Generates prompt variations through intelligent transformations
**Key Components**:
- Basic mutation operators (paraphrase, shorten, add constraints)
- Style transformation functions
- Context reduction algorithms

### 2. Template Engine
**Location**: `src/templates/`
**Purpose**: Structured prompt management and parameterization
**Key Components**:
- `PromptTemplate` interface for structured prompts
- Template parsing and serialization
- Parameter extraction and substitution

### 3. Configuration System
**Location**: `src/config/`
**Purpose**: Balance metrics and optimization parameters
**Key Components**:
- Balance metrics definitions (quality, cost, latency, reliability)
- Configurable weights and presets
- Validation functions for metric compliance

### 4. Evaluation Framework
**Location**: `evaluator.ts`
**Purpose**: Scoring and assessment of prompt variations
**Key Components**:
- Heuristic evaluation algorithms
- Performance measurement utilities
- Scoring aggregation and ranking

### 5. User Interface
**Location**: `prompt-engineer.tsx`
**Purpose**: Interactive prompt optimization interface
**Key Components**:
- React-based UI for prompt input
- Suggestion display and comparison
- Real-time optimization feedback

## Architectural Patterns

### Modular Design
- **Separation of Concerns**: Clear boundaries between mutation, evaluation, and presentation layers
- **Plugin Architecture**: Extensible mutation operators and evaluation metrics
- **Configuration-Driven**: Behavior controlled through configuration files rather than hard-coded values

### Data Flow Architecture
1. **Input Processing**: Raw prompts enter through UI or API
2. **Mutation Generation**: Multiple variations created through mutation operators
3. **Evaluation Pipeline**: Each variation scored across multiple dimensions
4. **Ranking & Selection**: Best candidates identified based on balance metrics
5. **Output Presentation**: Optimized suggestions presented to user

### TypeScript Integration
- **Strong Typing**: Comprehensive interfaces for all data structures
- **Type Safety**: Compile-time validation of prompt templates and configurations
- **IntelliSense Support**: Enhanced development experience with auto-completion

## Development Workflow
- **Test-Driven Development**: Unit tests in `src/__tests__/` directory
- **Configuration Management**: Example files demonstrate proper setup
- **Documentation-First**: Comprehensive planning documents guide implementation
- **Iterative Enhancement**: MVP approach with planned feature expansion