import React, { useState, useCallback } from 'react';
import { Sparkles, Copy, Check, Loader2, Zap, Target, Layers, ArrowRight } from 'lucide-react';

interface AnalysisResult {
  scratchpad: string;
  engineeredPrompt: string;
}

export default function PromptEngineer() {
  const [userQuery, setUserQuery] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const engineerPrompt = useCallback(async () => {
    if (!userQuery.trim()) return;
    
    setIsProcessing(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'claude-sonnet-4-20250514',
          max_tokens: 4000,
          messages: [
            {
              role: 'user',
              content: `You are an expert prompt engineer. Your task is to transform a user's query into an optimal, well-structured prompt that will elicit the best possible response from a language model.

Here is the user query to transform:

<user_query>
${userQuery}
</user_query>

First, analyze the query in a scratchpad. Consider:
- What is the user actually trying to accomplish?
- What information or context might be missing?
- What ambiguities need to be clarified?
- What output format would be most useful?
- Would examples help clarify the request?
- Are there any implicit assumptions that should be made explicit?

Then create an engineered prompt that:
- Preserves the core intent of the original query
- Adds necessary context and clarification
- Structures the request clearly
- Specifies the expected output format
- Includes helpful constraints or guidelines
- Uses XML tags or clear formatting where appropriate

Respond in this exact format:

<scratchpad>
[Your analysis here - be thorough but concise]
</scratchpad>

<engineered_prompt>
[The complete, standalone optimized prompt that could be given directly to an LLM]
</engineered_prompt>`
            }
          ]
        })
      });

      const data = await response.json();
      
      if (data.error) {
        throw new Error(data.error.message || 'API request failed');
      }

      const content = data.content?.[0]?.text || '';
      
      // Parse the response
      const scratchpadMatch = content.match(/<scratchpad>([\s\S]*?)<\/scratchpad>/);
      const promptMatch = content.match(/<engineered_prompt>([\s\S]*?)<\/engineered_prompt>/);
      
      if (scratchpadMatch && promptMatch) {
        setResult({
          scratchpad: scratchpadMatch[1].trim(),
          engineeredPrompt: promptMatch[1].trim()
        });
      } else {
        throw new Error('Could not parse the response');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsProcessing(false);
    }
  }, [userQuery]);

  const copyToClipboard = useCallback(async () => {
    if (result?.engineeredPrompt) {
      await navigator.clipboard.writeText(result.engineeredPrompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [result]);

  const exampleQueries = [
    "Write a blog post about AI",
    "Help me debug my code",
    "Explain quantum computing",
    "Create a marketing plan"
  ];

  return (
    <div className="min-h-screen bg-[#0a0a0f] text-[#e8e6e3] font-sans overflow-x-hidden">
      {/* Animated gradient background */}
      <div className="fixed inset-0 opacity-30">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-amber-500/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-orange-600/15 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute top-1/2 left-1/2 w-64 h-64 bg-yellow-500/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
      </div>

      {/* Grid overlay */}
      <div 
        className="fixed inset-0 opacity-[0.02]" 
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px'
        }}
      />

      <div className="relative z-10 max-w-6xl mx-auto px-6 py-12">
        {/* Header */}
        <header className="mb-16 text-center">
          <div className="inline-flex items-center gap-3 mb-6 px-4 py-2 bg-gradient-to-r from-amber-500/10 to-orange-500/10 border border-amber-500/20 rounded-full">
            <Sparkles className="w-4 h-4 text-amber-400" />
            <span className="text-xs uppercase tracking-[0.2em] text-amber-300/80">Prompt Engineering Studio</span>
          </div>
          
          <h1 className="text-5xl md:text-7xl font-extralight tracking-tight mb-4">
            <span className="bg-gradient-to-r from-amber-200 via-orange-200 to-amber-300 bg-clip-text text-transparent">
              Prompt
            </span>
            <span className="text-[#e8e6e3]/90"> Architect</span>
          </h1>
          
          <p className="text-lg text-[#e8e6e3]/50 font-light max-w-xl mx-auto">
            Transform ordinary queries into precisely engineered prompts that unlock superior AI responses
          </p>
        </header>

        {/* Features bar */}
        <div className="flex flex-wrap justify-center gap-8 mb-12">
          {[
            { icon: Target, label: 'Clarity Enhancement' },
            { icon: Layers, label: 'Structure Optimization' },
            { icon: Zap, label: 'Context Injection' }
          ].map(({ icon: Icon, label }) => (
            <div key={label} className="flex items-center gap-2 text-[#e8e6e3]/40">
              <Icon className="w-4 h-4" />
              <span className="text-xs uppercase tracking-wider">{label}</span>
            </div>
          ))}
        </div>

        {/* Main input section */}
        <div className="mb-8">
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-gradient-to-r from-amber-500/50 via-orange-500/50 to-amber-500/50 rounded-2xl blur opacity-20 group-hover:opacity-30 transition-opacity" />
            
            <div className="relative bg-[#12121a] border border-[#2a2a35] rounded-2xl p-6">
              <label className="block text-xs uppercase tracking-[0.15em] text-[#e8e6e3]/40 mb-3">
                Your Query
              </label>
              
              <textarea
                value={userQuery}
                onChange={(e) => setUserQuery(e.target.value)}
                placeholder="Enter your prompt to be optimized..."
                className="w-full h-40 bg-transparent text-[#e8e6e3] text-lg placeholder:text-[#e8e6e3]/20 resize-none focus:outline-none font-light leading-relaxed"
              />

              {/* Example chips */}
              <div className="flex flex-wrap gap-2 mt-4 pt-4 border-t border-[#2a2a35]">
                <span className="text-xs text-[#e8e6e3]/30 mr-2">Try:</span>
                {exampleQueries.map((example) => (
                  <button
                    key={example}
                    onClick={() => setUserQuery(example)}
                    className="px-3 py-1 text-xs bg-[#1a1a24] border border-[#2a2a35] rounded-full text-[#e8e6e3]/50 hover:text-amber-300/80 hover:border-amber-500/30 transition-all"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Transform button */}
          <div className="flex justify-center mt-8">
            <button
              onClick={engineerPrompt}
              disabled={isProcessing || !userQuery.trim()}
              className="group relative px-8 py-4 bg-gradient-to-r from-amber-500 to-orange-500 rounded-xl font-medium text-[#0a0a0f] disabled:opacity-50 disabled:cursor-not-allowed transition-all hover:shadow-lg hover:shadow-amber-500/25"
            >
              <span className="flex items-center gap-3">
                {isProcessing ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Engineering...
                  </>
                ) : (
                  <>
                    <Sparkles className="w-5 h-5" />
                    Engineer Prompt
                    <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                  </>
                )}
              </span>
            </button>
          </div>
        </div>

        {/* Error display */}
        {error && (
          <div className="mb-8 p-4 bg-red-500/10 border border-red-500/30 rounded-xl text-red-300 text-sm">
            {error}
          </div>
        )}

        {/* Results section */}
        {result && (
          <div className="space-y-8 animate-fadeIn">
            {/* Analysis section */}
            <div className="relative">
              <div className="absolute left-6 top-0 bottom-0 w-px bg-gradient-to-b from-amber-500/50 to-transparent" />
              
              <div className="pl-12">
                <h2 className="text-xs uppercase tracking-[0.2em] text-amber-400/70 mb-4 flex items-center gap-2">
                  <div className="w-2 h-2 bg-amber-400 rounded-full animate-pulse" />
                  Analysis
                </h2>
                
                <div className="bg-[#12121a]/80 border border-[#2a2a35] rounded-xl p-6">
                  <pre className="whitespace-pre-wrap text-sm text-[#e8e6e3]/60 font-mono leading-relaxed">
                    {result.scratchpad}
                  </pre>
                </div>
              </div>
            </div>

            {/* Engineered prompt section */}
            <div className="relative">
              <div className="absolute -inset-0.5 bg-gradient-to-r from-amber-500/30 to-orange-500/30 rounded-2xl blur opacity-30" />
              
              <div className="relative bg-[#12121a] border border-amber-500/30 rounded-2xl overflow-hidden">
                <div className="flex items-center justify-between px-6 py-4 border-b border-[#2a2a35] bg-gradient-to-r from-amber-500/5 to-transparent">
                  <h2 className="text-xs uppercase tracking-[0.2em] text-amber-300 flex items-center gap-2">
                    <Sparkles className="w-4 h-4" />
                    Engineered Prompt
                  </h2>
                  
                  <button
                    onClick={copyToClipboard}
                    className="flex items-center gap-2 px-3 py-1.5 bg-[#1a1a24] border border-[#2a2a35] rounded-lg text-xs text-[#e8e6e3]/70 hover:text-amber-300 hover:border-amber-500/30 transition-all"
                  >
                    {copied ? (
                      <>
                        <Check className="w-3 h-3" />
                        Copied!
                      </>
                    ) : (
                      <>
                        <Copy className="w-3 h-3" />
                        Copy
                      </>
                    )}
                  </button>
                </div>
                
                <div className="p-6">
                  <pre className="whitespace-pre-wrap text-[#e8e6e3]/90 font-mono text-sm leading-relaxed">
                    {result.engineeredPrompt}
                  </pre>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-20 text-center">
          <p className="text-xs text-[#e8e6e3]/20 tracking-wider">
            Powered by Claude • Transform queries into precision-engineered prompts
          </p>
        </footer>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400&display=swap');
        
        * {
          font-family: 'DM Sans', system-ui, sans-serif;
        }
        
        pre, code {
          font-family: 'JetBrains Mono', monospace;
        }
        
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: translateY(20px);
          }
          to {
            opacity: 1;
            transform: translateY(0);
          }
        }
        
        .animate-fadeIn {
          animation: fadeIn 0.6s ease-out forwards;
        }
      `}</style>
    </div>
  );
}
