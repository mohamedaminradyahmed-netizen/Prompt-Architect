/**
 * Unit Tests for SurrogateOrchestrator (DIRECTIVE-037)
 * 
 * Focus: Bug fixes and core functionality validation
 */

import { SurrogateOrchestrator, type EvaluationRequest } from '../../models/surrogateOrchestrator';

describe('SurrogateOrchestrator - Bug Fixes', () => {
  describe('Bug 1: getAlternativeModels should not include current model', () => {
    it('should exclude current model from alternatives', async () => {
      const orchestrator = new SurrogateOrchestrator({
        modeModelMap: {
          exploration: 'groq-llama-8b',
          exploitation: 'openai-gpt35',
          final: 'openai-gpt4',
        },
      });

      const request: EvaluationRequest = {
        prompt: 'Write a function',
      };

      // Test exploration mode (current: groq-llama-8b)
      const explorationSelection = await orchestrator.selectModel('exploration', request);
      
      expect(explorationSelection.model.model).toBe('llama-3.1-8b-instant');
      
      // Alternatives should NOT include groq-llama-8b
      const explorationAlternativeKeys = explorationSelection.alternativeModels.map(m => {
        // Find the registry key for this model
        const entry = Object.entries(orchestrator['modelRegistry']).find(
          ([_, config]) => config.model === m.model && config.provider === m.provider
        );
        return entry ? entry[0] : null;
      });
      
      expect(explorationAlternativeKeys).not.toContain('groq-llama-8b');

      // Test exploitation mode (current: openai-gpt35)
      const exploitationSelection = await orchestrator.selectModel('exploitation', request);
      
      const exploitationAlternativeKeys = exploitationSelection.alternativeModels.map(m => {
        const entry = Object.entries(orchestrator['modelRegistry']).find(
          ([_, config]) => config.model === m.model && config.provider === m.provider
        );
        return entry ? entry[0] : null;
      });
      
      expect(exploitationAlternativeKeys).not.toContain('openai-gpt35');
    });

    it('should return alternatives from same tier', async () => {
      const orchestrator = new SurrogateOrchestrator();
      const request: EvaluationRequest = { prompt: 'Test' };

      const selection = await orchestrator.selectModel('exploration', request);
      
      // All alternatives should be from 'cheap' tier
      selection.alternativeModels.forEach(model => {
        expect(model.tier).toBe('cheap');
      });
    });
  });
});
