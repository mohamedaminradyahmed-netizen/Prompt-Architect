# Prompt Architect - Development Guidelines

## Code Quality Standards

### Documentation Patterns
- **Comprehensive JSDoc Comments**: All modules start with detailed purpose descriptions and usage context
- **Section Separators**: Use `// ============================================================================` with descriptive headers to organize code sections
- **Interface Documentation**: Every interface includes detailed property descriptions with examples
- **Function Documentation**: Include purpose, parameters, return values, and usage examples

### File Organization Standards
- **Header Comments**: Every file begins with module purpose and overview
- **Logical Grouping**: Code organized into clear sections (INTERFACES, PRESETS, VALIDATION FUNCTIONS, HELPER FUNCTIONS, EXPORTS)
- **Export Organization**: Centralized exports at file end with both named and default exports
- **Import Grouping**: External dependencies first, then internal modules

### TypeScript Type Safety
- **Comprehensive Interfaces**: Strong typing for all data structures with detailed property descriptions
- **Union Types**: Use specific union types for constrained values (e.g., `'cost-optimized' | 'quality-first'`)
- **Generic Constraints**: Leverage TypeScript generics for flexible, reusable type definitions
- **Optional Properties**: Clear distinction between required and optional interface properties using `?` operator

## Structural Conventions

### Interface Design Patterns
- **Hierarchical Interfaces**: Complex types broken into logical sub-interfaces (e.g., `BalanceMetrics` contains `MetricWeights`)
- **Result Objects**: Standardized result patterns with `isValid`, `score`, `violations`, and `recommendation` properties
- **Metric Structures**: Consistent 0-1 scale for percentages, specific units for measurements (USD, milliseconds)

### Function Architecture
- **Pure Functions**: Stateless functions that return new objects rather than mutating inputs
- **Validation Functions**: Comprehensive validation with detailed error reporting and severity levels
- **Helper Functions**: Utility functions for common operations (normalization, comparison, calculation)
- **Factory Functions**: Constructor-like functions for creating configured objects (`createCustomMetrics`)

### Error Handling Patterns
- **Graceful Degradation**: Functions handle edge cases and return sensible defaults
- **Detailed Error Messages**: Human-readable error messages with specific values and thresholds
- **Severity Classification**: Three-tier severity system (low/medium/high) based on deviation percentages
- **Fallback Mechanisms**: Default behaviors when inputs are invalid or missing

## Semantic Patterns

### Configuration Management
- **Preset System**: Pre-configured objects for common use cases with descriptive names
- **Override Patterns**: Merge-based configuration allowing partial overrides of presets
- **Validation Utilities**: Built-in validation for configuration consistency (e.g., weight sum validation)
- **Normalization Functions**: Automatic correction of invalid configurations

### Scoring and Evaluation
- **Weighted Scoring**: Multi-dimensional scoring with configurable weights
- **Threshold Validation**: Clear pass/fail criteria with detailed violation reporting
- **Comparative Analysis**: Functions for ranking and comparing multiple candidates
- **Recommendation Generation**: Automated human-readable recommendations based on analysis

### Mutation System Architecture
- **Operator Pattern**: Individual mutation functions that transform input and return structured results
- **Variation Generation**: Systematic generation of multiple variations with duplicate prevention
- **Metadata Tracking**: Each mutation includes type information and transformation details
- **Randomization Control**: Controlled randomness with deterministic fallbacks

## Internal API Usage Patterns

### React Component Patterns
```typescript
// State management with proper typing
const [suggestions, setSuggestions] = useState<ScoredSuggestion[]>([]);

// Callback optimization with useCallback
const handleGenerate = useCallback(() => {
  // Implementation
}, [prompt]);

// Error handling with user feedback
const [error, setError] = useState<string | null>(null);
```

### Configuration API Usage
```typescript
// Preset usage
const metrics = getPreset('balanced');

// Custom configuration
const customMetrics = createCustomMetrics('balanced', {
  minQuality: 0.8,
  weights: { quality: 0.4, cost: 0.35, latency: 0.15, reliability: 0.1 }
});

// Validation integration
const result = validateMetrics(suggestionMetrics, metrics);
```

### Mutation API Patterns
```typescript
// Single mutation application
const variation = paraphraseMutation(originalPrompt);

// Batch variation generation
const variations = generateVariations(prompt, 4);

// Result structure consistency
return {
  prompt: transformedPrompt,
  mutation: 'mutation_type'
};
```

## Code Idioms and Conventions

### Array and Object Manipulation
- **Immutable Updates**: Use spread operator for object/array updates rather than mutation
- **Functional Array Methods**: Prefer `map`, `filter`, `reduce` over imperative loops
- **Optional Chaining**: Use `?.` for safe property access in uncertain contexts
- **Destructuring**: Extract properties cleanly in function parameters and assignments

### String Processing Patterns
- **Regular Expression Usage**: Complex pattern matching with descriptive variable names
- **Template Literals**: Multi-line strings and interpolation for readable text generation
- **Normalization**: Consistent whitespace and punctuation cleanup patterns
- **Length Calculations**: Percentage-based reduction tracking and validation

### Conditional Logic
- **Early Returns**: Validate inputs early and return immediately on invalid conditions
- **Ternary Operators**: Concise conditional assignments for simple cases
- **Switch Statements**: Not used; prefer object lookups or if-else chains for clarity
- **Null Coalescing**: Use `||` and `??` operators for default value assignment

### Performance Considerations
- **Memoization**: Use React hooks (`useCallback`, `useMemo`) for expensive operations
- **Lazy Evaluation**: Defer expensive calculations until needed
- **Batch Operations**: Process multiple items together rather than individually
- **Caching Patterns**: Store computed results to avoid recalculation

## Development Workflow Standards

### Testing Approach
- **Unit Test Structure**: Tests organized in `__tests__` directories
- **Mock Data**: Comprehensive example data for testing different scenarios
- **Edge Case Coverage**: Test boundary conditions and invalid inputs
- **Integration Examples**: Demonstrate proper API usage in example files

### Code Style Preferences
- **Consistent Indentation**: 2-space indentation throughout
- **Line Length**: Reasonable line lengths with logical breaking points
- **Naming Conventions**: Descriptive names that clearly indicate purpose and type
- **Comment Style**: Block comments for sections, inline comments for complex logic

### Module Design
- **Single Responsibility**: Each module has a clear, focused purpose
- **Dependency Management**: Minimal external dependencies, clear internal dependencies
- **Export Strategy**: Both named and default exports for flexibility
- **Configuration Separation**: Configuration isolated in dedicated modules