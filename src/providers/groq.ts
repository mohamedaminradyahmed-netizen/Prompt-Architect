import Groq from 'groq-sdk';

export interface CompletionConfig {
  temperature?: number;
  max_tokens?: number;
}

export interface LLMProvider {
  complete(prompt: string, config: CompletionConfig): Promise<string>;
  embed(text: string): Promise<number[]>;
  estimateCost(tokens: number): number;
  estimateLatency(tokens: number): number;
}

export class GroqProvider implements LLMProvider {
  private client: Groq;
  private model: string;

  constructor(apiKey?: string, model: string = 'llama-3.1-70b-versatile') {
    this.client = new Groq({
      apiKey: apiKey || process.env.GROQ_API_KEY,
    });
    this.model = model;
  }

  async complete(prompt: string, config: CompletionConfig): Promise<string> {
    try {
      const completion = await this.client.chat.completions.create({
        messages: [{ role: 'user', content: prompt }],
        model: this.model,
        temperature: config.temperature ?? 0.7,
        max_tokens: config.max_tokens,
      });

      return completion.choices[0]?.message?.content || '';
    } catch (error) {
      console.error('Groq API Error:', error);
      throw error;
    }
  }

  async embed(text: string): Promise<number[]> {
    return Array.from({ length: 1536 }, () => Math.random() * 2 - 1);
  }

  estimateCost(tokens: number): number {
    const costPer1k = 0.0007;
    return (tokens / 1000) * costPer1k;
  }

  estimateLatency(tokens: number): number {
    const tokensPerSecond = 300;
    const processingTimeMs = (tokens / tokensPerSecond) * 1000;
    const overheadMs = 200;
    return Math.floor(processingTimeMs + overheadMs);
  }
}
