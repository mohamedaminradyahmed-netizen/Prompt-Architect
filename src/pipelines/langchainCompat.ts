/**
 * LangChain Compatibility Layer (DIRECTIVE-043)
 *
 * Why:
 * - نريد Pipelines قابلة للتشغيل حتى لو لم يتم تثبيت LangChain بعد.
 * - عند توفر `@langchain/core` سيتم استخدام `RunnableSequence` مباشرة.
 * - عند عدم توفره نستخدم تنفيذ متسلسل بسيط يحافظ على نفس API: `invoke(input)`.
 */

export interface RunnableLike<I, O> {
  invoke(input: I): Promise<O>;
}

type StepFn = (input: any) => any | Promise<any>;

export async function createRunnableSequence<I, O>(
  steps: StepFn[]
): Promise<RunnableLike<I, O>> {
  // Prefer LangChain if available at runtime.
  try {
    const mod = (await import('@langchain/core/runnables')) as any;
    if (mod?.RunnableSequence?.from) {
      const seq = mod.RunnableSequence.from(steps);
      // Ensure we always return a Promise to match RunnableLike contract (and our fallback impl).
      return { invoke: async (input: I) => seq.invoke(input) };
    }
  } catch {
    // Ignore: LangChain is optional in this codebase right now.
  }

  // Fallback: run steps sequentially (same mental model as RunnableSequence).
  return {
    async invoke(input: I): Promise<O> {
      let current: any = input;
      for (const step of steps) current = await step(current);
      return current as O;
    },
  };
}

