# Prompt Refiner MVP Implementation

## Overview
Transform the current single-prompt engineer into a basic prompt refiner that generates multiple prompt variations, evaluates them, and displays 3 suggestions with scores and costs.

## Tasks

### 1. Create Mutation Operators
- [x] Implement paraphrase mutation: rephrase the prompt while preserving meaning
- [x] Implement shorten mutation: reduce prompt length while keeping key elements
- [x] Implement add constraint mutation: add helpful constraints or guidelines
- [x] Create a mutation utility module to apply these operators

### 2. Implement Evaluator Heuristics
- [x] Add token count calculation (simple word-based approximation)
- [x] Implement embedding similarity using cosine similarity (mock for now, can integrate real embeddings later)
- [x] Create scoring function combining token cost and similarity to original
- [x] Build evaluation utility module

### 3. Modify UI for Multiple Suggestions
- [x] Update result state to handle array of suggestions instead of single result
- [x] Change display to show 3 prompt variations in cards
- [x] Add score and cost display for each suggestion
- [x] Implement selection/copy functionality for individual suggestions

### 4. Integrate Mutations and Evaluation
- [ ] Modify engineerPrompt function to generate 3 variations using mutations
- [ ] Apply evaluation heuristics to score each variation
- [ ] Sort suggestions by score and select top 3
- [ ] Update API call to handle batch processing if needed

### 5. Testing and Polish
- [ ] Test with various input prompts
- [ ] Ensure UI handles edge cases (no results, errors)
- [ ] Optimize performance for multiple API calls
- [ ] Add loading states for batch processing
