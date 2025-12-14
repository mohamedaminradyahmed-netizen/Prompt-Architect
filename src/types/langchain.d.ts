// Minimal ambient declarations to keep TypeScript builds green when LangChain is not installed.
//
// Why: DIRECTIVE-043 يطلب استخدام LangChain، لكن المشروع قد يُبنى/يُختبر بدون هذه الاعتماديات.
// هذا الملف يسمح لنا بكتابة تكامل "اختياري" (Optional) دون فرض تثبيت حزم جديدة.
declare module '@langchain/core/runnables' {
  export const RunnableSequence: any;
}

declare module '@langchain/openai' {
  export const ChatOpenAI: any;
}

declare module 'langchain' {
  const anything: any;
  export default anything;
}
