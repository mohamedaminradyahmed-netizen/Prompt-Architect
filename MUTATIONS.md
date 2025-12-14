# Mutation Operators Documentation

## Available Mutations

### 1. paraphraseMutation
- **Purpose**: Rephrase the prompt while preserving meaning
- **Strategy**: Simple word substitutions and phrase transformations
- **Example**: "Write a function" → "Create a function"

### 2. shortenMutation  
- **Purpose**: Reduce prompt length while keeping key elements
- **Strategy**: Remove filler words and redundant phrases
- **Example**: "Please can you write a function" → "Write a function"

### 3. addConstraintMutation
- **Purpose**: Add helpful constraints or guidelines
- **Strategy**: Append structure and requirement suggestions
- **Example**: "Write a function" → "Write a function. Be specific and provide examples where appropriate."

### 4. reduceContextMutation
- **Purpose**: Reduces excessive context while preserving core meaning
- **Strategy**: Remove explanatory text, long examples, and inferable content
- **Target**: 30-50% length reduction
- **Example**: Long detailed prompt → Concise essential instructions

### 5. expandMutation ✨ NEW
- **Purpose**: Adds details and specificity to the prompt
- **Strategy**: Add definitions, steps, examples, and success criteria
- **Target**: 50-100% length increase with increased clarity
- **Features**:
  - Technical term definitions
  - Step-by-step instructions for general tasks
  - Illustrative examples when missing
  - Clear success criteria
  - Context enhancement for short prompts

#### expandMutation Examples:

**Input**: "Write a function"
**Output**: "Task: Write a function (step-by-step problem-solving procedure). Follow these steps: 1) Define the function signature, 2) Implement the core logic, 3) Add error handling, 4) Include documentation. For example, include input validation, return appropriate data types, and handle edge cases. Success criteria: The solution should be functional, well-documented, and follow best practices."

**Input**: "Optimize this code"  
**Output**: "Optimize (improve performance and efficiency) this code by: 1) Reducing time complexity, 2) Minimizing memory usage, 3) Improving readability. Measure performance before and after. For example, include input validation, return appropriate data types, and handle edge cases. Requirements: Ensure code is readable, maintainable, and properly tested."

## Usage

```typescript
import { expandMutation } from './mutations';

const result = expandMutation("Build a website");
console.log(result.prompt); // Expanded version with details
console.log(result.mutation); // "expand"
```

## Integration

The expandMutation is automatically included in:
- `applyRandomMutation()` - Random selection includes expand
- `generateVariations()` - Can generate expanded variations
- React UI - Will show "expand" mutation type in suggestions

## Implementation Status

✅ **DIRECTIVE-006 COMPLETED**
- Expand mutation operator implemented
- Integrated with existing mutation system
- Available in UI and evaluation pipeline
- Targets 50-100% length increase with clarity improvements