/**
 * Unit Tests for Mutation Operators
 *
 * Tests for DIRECTIVE-003: Try/Catch Style Mutation
 */

import { describe, it, expect } from '@jest/globals';
import { tryCatchStyleMutation, reduceContextMutation, PromptVariation } from '../mutations';

describe('Try/Catch Style Mutation (DIRECTIVE-003)', () => {
  describe('Basic Imperative Transformations', () => {
    it('should transform "Write a function" to try-style', () => {
      const input = 'Write a function to parse JSON';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to write');
      expect(result.text).toContain('parse JSON');
      expect(result.mutationType).toBe('try-catch-style');
      expect(result.expectedImpact.reliability).toBe('increase');
    });

    it('should transform "Create a component" to try-style', () => {
      const input = 'Create a React component for user authentication';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to create');
      expect(result.text).toContain('React component');
      expect(result.changeDescription).toContain('try/catch');
    });

    it('should transform "Build a system" to try-style', () => {
      const input = 'Build a distributed caching system';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to build');
      expect(result.text).toContain('distributed caching system');
    });
  });

  describe('Fix/Debug Transformations', () => {
    it('should transform "Fix the bug" with appropriate fallback', () => {
      const input = 'Fix the bug in the authentication system';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to fix');
      expect(result.text).toContain('authentication system');
      expect(result.text).toContain("can't fix it directly");
      expect(result.text).toContain('suggest possible solutions');
      expect(result.metadata?.transformationType).toBe('fix-with-fallback');
    });

    it('should transform "Debug the issue" with fallback', () => {
      const input = 'Debug the memory leak issue';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to debug');
      expect(result.text).toContain('memory leak');
      expect(result.text).toContain('workarounds');
    });

    it('should transform "Solve the problem" with fallback', () => {
      const input = 'Solve the performance problem in the database';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to solve');
      expect(result.text).toContain('performance problem');
    });
  });

  describe('Analysis/Review Transformations', () => {
    it('should transform "Analyze the code" with partial result option', () => {
      const input = 'Analyze the code for security vulnerabilities';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to analyze');
      expect(result.text).toContain('security vulnerabilities');
      expect(result.text).toContain('provide what you can determine');
      expect(result.metadata?.transformationType).toBe('analyze-with-partial');
    });

    it('should transform "Review the implementation" appropriately', () => {
      const input = 'Review the implementation for best practices';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to review');
      expect(result.text).toContain('best practices');
    });

    it('should transform "Examine the logs" with partial fallback', () => {
      const input = 'Examine the logs to find error patterns';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to examine');
      expect(result.text).toContain('error patterns');
    });
  });

  describe('General Transformations', () => {
    it('should transform "Explain the concept" appropriately', () => {
      const input = 'Explain the concept of closures in JavaScript';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to explain');
      expect(result.text).toContain('closures in JavaScript');
    });

    it('should transform "Optimize the function" appropriately', () => {
      const input = 'Optimize the sorting function';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to optimize');
      expect(result.text).toContain('sorting function');
    });

    it('should transform "Refactor the class" appropriately', () => {
      const input = 'Refactor the UserController class';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to refactor');
      expect(result.text).toContain('UserController class');
    });
  });

  describe('Complex Conditional Prompts', () => {
    it('should handle prompts with conditions', () => {
      const input = 'Write a function. If the input is invalid, throw an error. Make sure to handle edge cases.';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to');
      expect(result.text.length).toBeGreaterThan(input.length);
    });

    it('should break down complex instructions', () => {
      const input = 'Implement authentication. Use JWT tokens. Store sessions in Redis.';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to');
      expect(result.mutationType).toBe('try-catch-style');
    });
  });

  describe('Non-Imperative Prompts', () => {
    it('should handle prompts without clear imperative verbs', () => {
      const input = 'What are the best practices for React performance?';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to');
      expect(result.text).toContain('best practices');
      expect(result.metadata?.transformationType).toBe('general-wrapper');
    });

    it('should handle descriptive prompts', () => {
      const input = 'The system needs better error handling';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to');
      expect(result.text).toContain('error handling');
    });
  });

  describe('Metadata and Impact', () => {
    it('should provide accurate metadata', () => {
      const input = 'Write a function';
      const result = tryCatchStyleMutation(input);

      expect(result.metadata).toBeDefined();
      expect(result.metadata?.originalLength).toBe(input.length);
      expect(result.metadata?.newLength).toBe(result.text.length);
      expect(result.metadata?.lengthIncrease).toBeGreaterThan(0);
    });

    it('should indicate expected impact on metrics', () => {
      const input = 'Fix the bug';
      const result = tryCatchStyleMutation(input);

      expect(result.expectedImpact).toBeDefined();
      expect(result.expectedImpact.reliability).toBe('increase');
      expect(result.expectedImpact.cost).toBe('increase'); // Longer prompt
    });

    it('should track transformation type', () => {
      const fixPrompt = 'Fix the authentication bug';
      const createPrompt = 'Create a new API endpoint';
      const analyzePrompt = 'Analyze the performance metrics';

      expect(tryCatchStyleMutation(fixPrompt).metadata?.transformationType).toBe('fix-with-fallback');
      expect(tryCatchStyleMutation(createPrompt).metadata?.transformationType).toBe('create-with-alternatives');
      expect(tryCatchStyleMutation(analyzePrompt).metadata?.transformationType).toBe('analyze-with-partial');
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty prompts', () => {
      const input = '';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toBeDefined();
      expect(result.mutationType).toBe('try-catch-style');
    });

    it('should handle very short prompts', () => {
      const input = 'Help';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to');
    });

    it('should handle prompts with special characters', () => {
      const input = 'Write a function to parse JSON with { nested: "objects" }';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to write');
      expect(result.text).toContain('{ nested: "objects" }');
    });

    it('should preserve multiline prompts', () => {
      const input = 'Write a function\nthat handles\nmultiple lines';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to');
    });
  });

  describe('Real-World Examples', () => {
    it('should handle code generation request', () => {
      const input = 'Write a TypeScript function that validates email addresses using regex';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to write a typescript function');
      expect(result.text).toContain('validates email addresses');
      expect(result.text).toContain('If you encounter issues, suggest alternatives');
    });

    it('should handle bug fix request', () => {
      const input = 'Fix the race condition in the user registration process';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to fix');
      expect(result.text).toContain('race condition');
      expect(result.text).toContain("can't fix it directly");
      expect(result.text).toContain('suggest possible solutions');
    });

    it('should handle code review request', () => {
      const input = 'Review this pull request for security issues and performance problems';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to review');
      expect(result.text).toContain('security issues');
      expect(result.text).toContain('performance problems');
    });

    it('should handle optimization request', () => {
      const input = 'Optimize the database queries in the user service';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('Try to optimize');
      expect(result.text).toContain('database queries');
      expect(result.text).toContain('user service');
    });
  });

  describe('Preservation of Original Meaning', () => {
    it('should preserve the core instruction', () => {
      const input = 'Write a function to calculate Fibonacci numbers';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('calculate Fibonacci numbers');
    });

    it('should preserve technical terms', () => {
      const input = 'Implement a binary search tree with O(log n) insertion';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('binary search tree');
      expect(result.text).toContain('O(log n)');
      expect(result.text).toContain('insertion');
    });

    it('should preserve constraints and requirements', () => {
      const input = 'Create a REST API endpoint with proper error handling and authentication';
      const result = tryCatchStyleMutation(input);

      expect(result.text).toContain('REST API endpoint');
      expect(result.text).toContain('error handling');
      expect(result.text).toContain('authentication');
    });
  });
});

// ============================================================================
// CONTEXT REDUCTION MUTATION TESTS (DIRECTIVE-004)
// ============================================================================

describe('Context Reduction Mutation (DIRECTIVE-004)', () => {
  describe('Explanatory Phrase Removal', () => {
    it('should remove "in other words" explanatory phrases', () => {
      const input = 'Write a function to sort an array. In other words, arrange the elements in ascending order.';
      const result = reduceContextMutation(input);

      expect(result.text).not.toContain('In other words');
      expect(result.text).toContain('Write a function');
      expect(result.mutationType).toBe('context-reduction');
    });

    it('should remove "basically" filler phrases', () => {
      const input = 'Create a user registration form. Basically, it should collect name, email, and password.';
      const result = reduceContextMutation(input);

      expect(result.text).not.toContain('Basically');
      expect(result.text).toContain('registration form');
    });

    it('should remove "to put it simply" phrases', () => {
      const input = 'Implement a caching system. To put it simply, store frequently accessed data in memory.';
      const result = reduceContextMutation(input);

      expect(result.text).not.toContain('To put it simply');
      expect(result.text).toContain('caching system');
    });
  });

  describe('Long Example Replacement', () => {
    it('should replace long "for example" blocks with brief reference', () => {
      const input = 'Write a parser. For example, you could use a recursive descent parser that handles nested structures by recursively calling the parse function for each level of nesting.';
      const result = reduceContextMutation(input);

      expect(result.text.length).toBeLessThan(input.length);
      expect(result.text).toContain('Write a parser');
    });

    it('should replace long "such as" blocks', () => {
      const input = 'Support multiple data formats, such as JSON which uses braces and brackets for structure, XML which uses opening and closing tags, and YAML which uses indentation for nesting.';
      const result = reduceContextMutation(input);

      expect(result.text.length).toBeLessThan(input.length);
    });
  });

  describe('Inferable Content Removal', () => {
    it('should remove "as you may know" phrases', () => {
      const input = 'As you may know, TypeScript is a typed superset of JavaScript. Write a TypeScript function.';
      const result = reduceContextMutation(input);

      expect(result.text).not.toContain('As you may know');
      expect(result.text).toContain('TypeScript');
    });

    it('should remove "obviously" and similar words', () => {
      const input = 'Obviously, the function should handle errors. Create error handling logic.';
      const result = reduceContextMutation(input);

      expect(result.text).not.toContain('Obviously');
      expect(result.text).toContain('error');
    });

    it('should remove "needless to say" phrases', () => {
      const input = 'Needless to say, security is important. Implement authentication.';
      const result = reduceContextMutation(input);

      expect(result.text).not.toContain('Needless to say');
      expect(result.text).toContain('authentication');
    });

    it('should remove "of course" phrases', () => {
      const input = 'Of course, you need to validate input. Create a validation function.';
      const result = reduceContextMutation(input);

      expect(result.text).not.toContain('Of course');
      expect(result.text).toContain('validation');
    });
  });

  describe('Redundant Introduction Removal', () => {
    it('should remove "I would like you to"', () => {
      const input = 'I would like you to write a function that calculates prime numbers';
      const result = reduceContextMutation(input);

      expect(result.text).not.toContain('I would like you to');
      expect(result.text).toContain('write a function');
    });

    it('should remove "I want you to"', () => {
      const input = 'I want you to create a database schema for users';
      const result = reduceContextMutation(input);

      expect(result.text).not.toContain('I want you to');
      expect(result.text).toContain('create a database');
    });

    it('should remove "I need you to"', () => {
      const input = 'I need you to fix the bug in the authentication module';
      const result = reduceContextMutation(input);

      expect(result.text).not.toContain('I need you to');
      expect(result.text).toContain('fix the bug');
    });

    it('should convert "It is essential that" to "Must"', () => {
      const input = 'It is essential that the code handles all edge cases';
      const result = reduceContextMutation(input);

      expect(result.text).toContain('Must');
      expect(result.text).not.toContain('It is essential that');
    });
  });

  describe('Preservation of Core Content', () => {
    it('should preserve technical constraints', () => {
      const input = 'Write a function using TypeScript. It must be async/await compatible. Obviously, this is for Node.js.';
      const result = reduceContextMutation(input);

      expect(result.text).toContain('TypeScript');
      expect(result.text).toContain('async/await');
    });

    it('should preserve action verbs', () => {
      const input = 'I would like you to create a component that renders a list of items';
      const result = reduceContextMutation(input);

      expect(result.text).toContain('create');
      expect(result.text).toContain('renders');
      expect(result.text).toContain('list');
    });

    it('should preserve important requirements', () => {
      const input = 'Build an API. It must handle 1000 requests per second. Basically, it needs to be scalable.';
      const result = reduceContextMutation(input);

      expect(result.text).toContain('1000 requests');
      expect(result.text).toContain('Build an API');
    });
  });

  describe('Length Reduction Metrics', () => {
    it('should achieve some reduction on verbose prompts', () => {
      const input = 'I would like you to write a function. As you know, functions are reusable blocks of code. Obviously, this function should be well-documented. Basically, just create something that works.';
      const result = reduceContextMutation(input);

      expect(result.metadata?.originalLength).toBe(input.length);
      expect(result.metadata?.newLength).toBeLessThan(input.length);
      expect(result.metadata?.reductionPercent).toBeGreaterThan(0);
    });

    it('should not reduce already concise prompts significantly', () => {
      const input = 'Write a sorting function';
      const result = reduceContextMutation(input);

      expect(result.text).toContain('sorting function');
      expect(result.text.length).toBeGreaterThan(5);
    });

    it('should provide expected impact metrics', () => {
      const input = 'I would like you to create a function';
      const result = reduceContextMutation(input);

      expect(result.expectedImpact.cost).toBe('decrease');
      expect(result.expectedImpact.latency).toBe('decrease');
      expect(result.expectedImpact.reliability).toBe('neutral');
    });
  });

  describe('Edge Cases', () => {
    it('should handle empty prompts gracefully', () => {
      const input = '';
      const result = reduceContextMutation(input);

      expect(result.text).toBeDefined();
      expect(result.mutationType).toBe('context-reduction');
    });

    it('should handle very short prompts', () => {
      const input = 'Write code';
      const result = reduceContextMutation(input);

      expect(result.text.length).toBeGreaterThan(0);
    });

    it('should handle prompts with special characters', () => {
      const input = 'I would like you to parse JSON with { "key": "value" } format';
      const result = reduceContextMutation(input);

      expect(result.text).toContain('{ "key": "value" }');
    });

    it('should handle multiline prompts', () => {
      const input = 'Write a function.\nObviously it should be clean.\nBasically just make it work.';
      const result = reduceContextMutation(input);

      expect(result.text).not.toContain('Obviously');
      expect(result.text).not.toContain('Basically');
    });
  });

  describe('Real-World Examples', () => {
    it('should reduce a verbose code request', () => {
      const input = 'I would like you to write a TypeScript function. As you probably know, TypeScript adds types to JavaScript. In other words, it helps catch errors early. Basically, I need a function to validate email addresses.';
      const result = reduceContextMutation(input);

      expect(result.text).toContain('TypeScript');
      expect(result.text).toContain('validate email');
      expect(result.metadata?.reductionPercent).toBeGreaterThan(0);
    });

    it('should reduce a verbose explanation request', () => {
      const input = 'Obviously, React is popular. As we all know, it uses a virtual DOM. I need you to explain how the virtual DOM works. Needless to say, this is for my learning.';
      const result = reduceContextMutation(input);

      expect(result.text).toContain('virtual DOM');
      expect(result.text).toContain('explain');
      expect(result.text).not.toContain('Obviously');
      expect(result.text).not.toContain('Needless to say');
    });
  });
});
