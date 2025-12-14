# أوامر توجيهية لتطوير نظام Prompt Refiner

# Directives for AI Coding Agent

> **السياق العام**: أنت تعمل على تطوير نظام **Prompt Refiner** متقدم يقوم بتحسين البرومبتات تلقائياً لتحقيق توازن بين الجودة والتكلفة والزمن والموثوقية. المشروع مبني بـ TypeScript + React وحالياً في مرحلة MVP الأساسية.

> **الملفات الرئيسية الحالية**:
>
> - `mutations.ts` - يحتوي على 3 mutation operators أساسية
> - `evaluator.ts` - يحتوي على نظام تقييم heuristic بسيط
> - `prompt-engineer.tsx` - واجهة React تعرض 3 اقتراحات
> - `TODO.md` - قائمة المهام الكاملة
> - `PLAN.md` - الخطة الاستراتيجية

---

## 🎯 المرحلة 0: الأساسيات والمعايير (الأولوية القصوى)

### DIRECTIVE-001: تحديد معايير التوازن

```
المهمة: قم بإنشاء ملف `config/balanceMetrics.ts` يحدد معايير التوازن بين:
- الدقة/الجودة (Accuracy/Quality): ما هو الحد الأدنى المقبول؟
- التكلفة (Cost): ما هو سقف التكلفة المستهدف لكل طلب؟
- الزمن (Latency): ما هو الحد الأقصى المقبول للاستجابة؟
- الموثوقية (Reliability): ما هو معدل الهلوسة المقبول؟

المخرجات المطلوبة:
1. interface `BalanceMetrics` يحدد جميع المعايير
2. وظيفة `validateMetrics()` للتحقق من أن الاقتراح يلبي المعايير
3. أوزان قابلة للتخصيص لكل معيار (configurable weights)
4. presets جاهزة: "cost-optimized", "quality-first", "balanced"

الملف: `src/config/balanceMetrics.ts`
```

### DIRECTIVE-002: تصنيف أنواع البرومبتات

```
المهمة: قم بإنشاء نظام تصنيف للبرومبتات في `src/types/promptTypes.ts`

المخرجات المطلوبة:
1. enum `PromptCategory` يشمل:
   - CODE_GENERATION
   - CODE_REVIEW
   - CONTENT_WRITING
   - MARKETING_COPY
   - DATA_ANALYSIS
   - GENERAL_QA
   - CREATIVE_WRITING

2. interface `PromptClassification` مع:
   - category: PromptCategory
   - confidence: number (0-1)
   - characteristics: string[]

3. وظيفة `classifyPrompt(prompt: string): PromptClassification`
   - تحلل النص وتحدد الفئة تلقائياً
   - تستخدم keyword matching و pattern recognition

4. مقاييس مخصصة لكل فئة (category-specific metrics)

الملف: `src/types/promptTypes.ts`
التبعيات: لا يوجد
```

---

## 🧩 المرحلة 1: تطوير Mutation Operators المتقدمة

### DIRECTIVE-003: تطوير Try/Catch Style Mutation

```
المهمة: أضف mutation operator جديد في `mutations.ts` يحول أسلوب التعليمات

الوظيفة المطلوبة:
export function tryCatchStyleMutation(prompt: string): PromptVariation

المنطق:
- إذا كان البرومبت يحتوي على تعليمات مباشرة (imperative)
  → حوّله لأسلوب "حاول أن..." (Try to...)
- إذا كان يحتوي على شروط معقدة
  → قسّمه إلى: "Try X. If that fails, try Y."

أمثلة:
- "Write a function" → "Try to write a function that..."
- "Fix the bug in X" → "Try to identify and fix the bug. If you can't, suggest alternatives."

الملف: `src/mutations.ts`
الاختبار: أضف unit tests في `src/__tests__/mutations.test.ts`
```

### DIRECTIVE-004: تطوير Context Reduction Mutation

```
المهمة: أضف mutation operator يقلل السياق الزائد

الوظيفة المطلوبة:
export function reduceContextMutation(prompt: string): PromptVariation

المنطق:
1. حدد الجمل الثانوية أو التفسيرية
2. احتفظ فقط بالتعليمات الأساسية
3. أزل الأمثلة الطويلة واستبدلها بإشارات مختصرة
4. أزل الشروح التي يمكن استنتاجها

قواعد:
- احتفظ بجميع القيود (constraints) الأساسية
- لا تزل معلومات تقنية مهمة
- استهدف تقليل 30-50% من الطول

الملف: `src/mutations.ts`
الاختبار: تأكد من أن المخرج يحتفظ بالمعنى الأساسي
```

### DIRECTIVE-005: بناء نظام Parameterized Templates [DONE]

```
المهمة: أنشئ نظام قوالب مُهيكلة في `src/templates/`

البنية المطلوبة:
interface PromptTemplate {
  role?: string;           // "You are a senior software engineer"
  goal: string;            // "Write a function that..."
  constraints?: string[];  // ["Must be in TypeScript", "Use async/await"]
  examples?: string[];     // ["Example 1: ...", "Example 2: ..."]
  format?: string;         // "Return as JSON", "Use markdown"
}

الوظائف المطلوبة:
1. `parsePromptToTemplate(prompt: string): PromptTemplate`
   - تحلل prompt حر وتستخرج المكونات

2. `templateToPrompt(template: PromptTemplate): string`
   - تحول Template إلى نص برومبت منسق

3. `mutateTemplate(template: PromptTemplate, mutation: string): PromptTemplate`
   - تطبق تعديلات على مستوى Template

الملفات:
- `src/templates/PromptTemplate.ts`
- `src/templates/templateParser.ts`
- `src/templates/templateMutations.ts`
```

### DIRECTIVE-006: تطوير Expand Mutation

```
المهمة: أضف mutation operator يوسّع البرومبت بتفاصيل إضافية

الوظيفة المطلوبة:
export function expandMutation(prompt: string): PromptVariation

المنطق:
1. حدد المصطلحات التقنية → أضف تعريفات مختصرة
2. حدد التعليمات العامة → أضف خطوات محددة
3. أضف أمثلة توضيحية إن لم تكن موجودة
4. أضف معايير نجاح واضحة

أمثلة:
- "Optimize this code"
  → "Optimize this code by: 1) Reducing time complexity, 2) Minimizing memory usage, 3) Improving readability. Measure performance before and after."

الملف: `src/mutations.ts`
الهدف: زيادة 50-100% في الطول مع زيادة الوضوح
```

### DIRECTIVE-007: تطوير Constrain Mutation

```
المهمة: أضف mutation operator يضيف قيوداً محددة

الوظيفة المطلوبة:
export function constrainMutation(prompt: string, category: PromptCategory): PromptVariation

القيود حسب الفئة:
- CODE_GENERATION: "Use TypeScript", "Add error handling", "Include unit tests"
- CONTENT_WRITING: "Max 500 words", "Use active voice", "Grade level 8"
- MARKETING_COPY: "Include CTA", "Focus on benefits", "Use emotional triggers"

المنطق:
1. حدد فئة البرومبت
2. اختر 2-3 قيود مناسبة من مكتبة القيود
3. أضفها بطريقة طبيعية للبرومبت

الملفات:
- `src/mutations.ts` - الوظيفة الرئيسية
- `src/constraints/constraintLibrary.ts` - مكتبة القيود
```

### DIRECTIVE-008: تطوير Task Decomposition Strategy

```
المهمة: أنشئ نظام تقسيم المهام المعقدة لمهام فرعية

الوظيفة المطلوبة:
export function decomposeTaskMutation(prompt: string): PromptVariation[]

المنطق:
1. حلل البرومبت وحدد إذا كان يحتوي على مهام متعددة
2. قسّم إلى sub-prompts منفصلة
3. أضف تعليمات ربط بين المهام
4. أنشئ "orchestration prompt" يجمع النتائج

أمثلة:
Input: "Build a user authentication system with email verification"
Output:
- Prompt 1: "Design database schema for user authentication"
- Prompt 2: "Implement user registration endpoint"
- Prompt 3: "Create email verification system"
- Orchestrator: "Integrate the three components into a complete auth system"

الملف: `src/strategies/taskDecomposition.ts`
```

### DIRECTIVE-009: تطوير Multi-Step Prompts System

```
المهمة: أنشئ نظام prompts متعددة الخطوات

البنية المطلوبة:
interface MultiStepPrompt {
  steps: PromptStep[];
  dependencies: Map<number, number[]>;  // step -> depends on steps
  aggregationStrategy: 'sequential' | 'parallel' | 'conditional';
}

interface PromptStep {
  id: number;
  prompt: string;
  expectedOutputType: 'code' | 'text' | 'json' | 'analysis';
  validation?: (output: string) => boolean;
}

الوظائف المطلوبة:
1. `createMultiStepPrompt(originalPrompt: string): MultiStepPrompt`
2. `executeMultiStep(multiStep: MultiStepPrompt, executor: LLMExecutor): Promise<string>`
3. `validateStep(step: PromptStep, output: string): boolean`

الملف: `src/strategies/multiStep.ts`
```

---

## 📊 المرحلة 2: تحسين نظام التقييم (Evaluation System)

### DIRECTIVE-010: إضافة قياس Latency الفعلي

```
المهمة: أضف قياساً فعلياً لزمن الاستجابة

المطلوب في `evaluator.ts`:
1. أضف حقل `latency: number` في `ScoredSuggestion`
2. أنشئ وظيفة:
   async function measureLatency(
     prompt: string,
     provider: 'openai' | 'anthropic' | 'groq'
   ): Promise<number>

3. قم بقياس:
   - Time to first token (TTFT)
   - Total response time
   - Network latency

4. احفظ القياسات في cache للاستخدام المستقبلي

الملف: `src/evaluator.ts`
التبعيات: ستحتاج إلى API clients للـ LLM providers
```

### DIRECTIVE-011: قياس طول الإخراج الفعلي

```
المهمة: أضف قياساً فعلياً لطول المخرجات

الوظيفة المطلوبة:
async function measureActualOutput(
  prompt: string,
  provider: LLMProvider,
  samples: number = 3
): Promise<OutputMetrics>

interface OutputMetrics {
  avgLength: number;
  avgTokens: number;
  variance: number;
  quality: number;  // سيتم تحديده لاحقاً
}

المنطق:
1. قم بتشغيل البرومبت عدة مرات (samples)
2. قس طول كل مخرج
3. احسب المتوسط والتباين
4. خزّن النتائج مع timestamp

الملف: `src/evaluator.ts`
الاختبار: اختبر مع prompts مختلفة
```

### DIRECTIVE-012: دمج ROUGE/BLEU Metrics

```
المهمة: أضف دعماً لمقاييس ROUGE و BLEU للمخرجات المرجعية

التثبيت المطلوب:
npm install rouge-score bleu-score

الوظائف المطلوبة:
1. `calculateROUGE(candidate: string, reference: string): ROUGEScores`
   - يحسب ROUGE-1, ROUGE-2, ROUGE-L

2. `calculateBLEU(candidate: string, references: string[]): number`
   - يدعم multiple references

3. `evaluateAgainstReference(
     prompt: string,
     output: string,
     referenceOutputs: string[]
   ): ReferenceMetrics`

الملف: `src/evaluator/referenceMetrics.ts`
الاستخدام: فقط عندما يوجد مخرجات مرجعية
```

### DIRECTIVE-013: بناء نظام كشف الهلوسة

```
المهمة: أنشئ نظام للكشف عن الهلوسة في المخرجات

الاستراتيجيات المطلوبة:
1. **Consistency Check**: شغّل البرومبت مرتين واقارن المخرجات
2. **Fact Verification**: تحقق من الحقائق القابلة للتحقق
3. **Confidence Scoring**: استخدم logprobs للكشف عن عدم اليقين

الوظيفة المطلوبة:
async function detectHallucination(
  prompt: string,
  output: string,
  context?: string
): Promise<HallucinationScore>

interface HallucinationScore {
  score: number;              // 0-1, أعلى = أكثر احتمالية للهلوسة
  confidence: number;         // ثقة النموذج في التصنيف
  inconsistencies: string[];  // قائمة بالتناقضات المكتشفة
  method: string;             // الطريقة المستخدمة للكشف
}

الملف: `src/evaluator/hallucinationDetector.ts`
```

### DIRECTIVE-014: بناء فحص Factuality عبر RAG

```
المهمة: أنشئ نظام RAG للتحقق من صحة الحقائق

المكونات المطلوبة:
1. Vector Database Setup (Pinecone أو Weaviate)
2. Knowledge Base من مصادر موثوقة
3. Retrieval System للحقائق ذات الصلة
4. Verification Logic

الوظيفة المطلوبة:
async function verifyFactuality(
  claim: string,
  context?: string
): Promise<FactualityCheck>

interface FactualityCheck {
  isFactual: boolean;
  confidence: number;
  sources: string[];          // مصادر داعمة
  contradictions: string[];   // معلومات متناقضة
}

الملفات:
- `src/rag/vectorStore.ts`
- `src/rag/retrieval.ts`
- `src/evaluator/factualityChecker.ts`

التبعيات: تحتاج إلى vector DB و embedding model
```

### DIRECTIVE-015: بناء نظام Human Feedback Score

```
المهمة: أنشئ نظام لجمع وإدارة تقييمات البشر

قاعدة البيانات المطلوبة:
CREATE TABLE human_feedback (
  id SERIAL PRIMARY KEY,
  prompt_id VARCHAR,
  variation_id VARCHAR,
  score INT CHECK (score >= 1 AND score <= 5),
  feedback_text TEXT,
  user_id VARCHAR,
  timestamp TIMESTAMP,
  metadata JSONB
);

الواجهة المطلوبة في UI:
- نظام تقييم 5 نجوم لكل اقتراح
- حقل نص اختياري للملاحظات
- أزرار سريعة: "Perfect", "Good", "Needs Work", "Poor"

الـ Backend المطلوب:
1. API endpoint: POST /api/feedback
2. وظيفة `storeFeedback(feedback: HumanFeedback): Promise<void>`
3. وظيفة `getAverageFeedback(variationId: string): Promise<number>`
4. Dashboard لعرض إحصائيات الـ feedback

الملفات:
- `src/api/feedback.ts`
- `src/db/feedbackStore.ts`
- `src/components/FeedbackWidget.tsx`
```

### DIRECTIVE-016: مقاييس مخصصة للكود

```
المهمة: طوّر مقاييس خاصة بتقييم جودة الكود المُولّد

المقاييس المطلوبة:
1. **Syntax Correctness**: هل الكود يمكن تشغيله؟
2. **Best Practices**: هل يتبع معايير البرمجة؟
3. **Test Coverage**: هل يتضمن اختبارات؟
4. **Documentation**: هل موثّق بشكل كافٍ؟
5. **Security**: هل يحتوي على ثغرات أمنية؟
6. **Performance**: هل الكود مُحسّن؟

الوظيفة المطلوبة:
async function evaluateCodeQuality(
  code: string,
  language: string
): Promise<CodeQualityMetrics>

interface CodeQualityMetrics {
  syntaxScore: number;      // 0-100
  bestPracticesScore: number;
  hasTests: boolean;
  documentationScore: number;
  securityIssues: SecurityIssue[];
  performanceScore: number;
  overallScore: number;
}

الأدوات المطلوبة:
- استخدم ESLint/TSLint للـ linting
- استخدم static analysis tools
- قم بتشغيل الكود في sandbox للتحقق من الأخطاء

الملف: `src/evaluator/codeQualityEvaluator.ts`
```

### DIRECTIVE-017: مقاييس مخصصة للتسويق/المحتوى

```
المهمة: طوّر مقاييس خاصة بتقييم جودة المحتوى التسويقي

المقاييس المطلوبة:
1. **Tone Consistency**: هل النبرة متسقة مع Brand Voice؟
2. **Readability**: مستوى القراءة (Flesch-Kincaid)
3. **SEO Score**: جودة SEO (keywords, meta, structure)
4. **Engagement Potential**: احتمالية التفاعل
5. **Call-to-Action**: وجود وفعالية CTA
6. **Emotional Appeal**: قوة الجاذبية العاطفية

الوظيفة المطلوبة:
function evaluateContentQuality(
  content: string,
  targetAudience?: string,
  brandVoice?: BrandVoice
): ContentQualityMetrics

interface ContentQualityMetrics {
  toneScore: number;
  readabilityScore: number;     // Flesch Reading Ease
  gradeLevel: number;           // Flesch-Kincaid Grade
  seoScore: number;
  hasCTA: boolean;
  ctaEffectiveness: number;
  emotionalScore: number;
  overallScore: number;
}

المكتبات المطلوبة:
- flesch-kincaid
- sentiment analysis library
- keyword density calculator

الملف: `src/evaluator/contentQualityEvaluator.ts`
```

### DIRECTIVE-018: تحسين Similarity باستخدام Embeddings حقيقية

```
المهمة: استبدل word frequency similarity بـ embeddings حقيقية

المطلوب:
1. دمج OpenAI Embeddings API أو sentence-transformers
2. استبدال وظيفة `calculateSimilarity()` الحالية
3. إضافة caching للـ embeddings لتقليل التكلفة

الوظيفة الجديدة:
async function calculateSemanticSimilarity(
  text1: string,
  text2: string,
  useCache: boolean = true
): Promise<number>

الخطوات:
1. تحويل النصوص إلى embeddings
2. حساب cosine similarity بين الـ vectors
3. تخزين النتائج في cache
4. إرجاع القيمة (0-1)

الملف: `src/evaluator.ts`
التبعيات: openai package أو @xenova/transformers
التكلفة: راقب استهلاك API calls

قبل:
// Simple word frequency
calculateSimilarity(text1, text2)

بعد:
// Semantic embeddings
await calculateSemanticSimilarity(text1, text2)
```

---

## 🔧 المرحلة 3: بناء Optimizer (خوارزميات التحسين)

### DIRECTIVE-019: تنفيذ Hill-Climbing Optimizer [DONE]

```
المهمة: أنشئ optimizer بسيط يستخدم Hill-Climbing

المبدأ:
1. ابدأ من prompt أصلي
2. طبّق mutation عشوائي
3. قيّم النتيجة
4. إذا كان أفضل، احتفظ به واستمر
5. إذا لم يكن أفضل، تراجع وجرّب mutation آخر
6. توقف عند عدد محدد من الخطوات أو عندما لا يوجد تحسن

الوظيفة المطلوبة:
async function hillClimbingOptimize(
  initialPrompt: string,
  maxIterations: number = 10,
  scoringFunction: ScoringFunction
): Promise<OptimizationResult>

interface OptimizationResult {
  bestPrompt: string;
  bestScore: number;
  iterations: number;
  history: {prompt: string, score: number}[];
}

الملف: `src/optimizer/hillClimbing.ts`
الاختبار: جرّب على عدة prompts واعرض التحسن
```

### DIRECTIVE-020: تنفيذ Genetic/Population-based Optimizer [DONE]

```
المهمة: أنشئ optimizer يستخدم Genetic Algorithm

الخطوات:
1. **Initialize Population**: أنشئ 20 variation عشوائية
2. **Evaluate**: قيّم كل variation
3. **Selection**: اختر أفضل 10 (top 50%)
4. **Crossover**: امزج بين الأفضل لإنشاء أطفال جدد
5. **Mutation**: طبّق mutations عشوائية
6. **Repeat**: كرر لعدد من الأجيال

الوظيفة المطلوبة:
async function geneticOptimize(
  initialPrompt: string,
  config: GeneticConfig
): Promise<PopulationResult>

interface GeneticConfig {
  populationSize: number;      // عدد الأفراد في كل جيل
  generations: number;         // عدد الأجيال
  crossoverRate: number;       // 0-1
  mutationRate: number;        // 0-1
  elitismCount: number;        // عدد الأفضل للاحتفاظ بهم
}

interface PopulationResult {
  bestPrompts: string[];       // أفضل 5 prompts
  scores: number[];
  generationHistory: Generation[];
}

الملف: `src/optimizer/genetic.ts`
```

### DIRECTIVE-021: تنفيذ Bayesian Optimization [DONE]

```
المهمة: أنشئ optimizer للمعلمات باستخدام Bayesian Optimization

الاستخدام: تحسين معلمات Template (role, constraints, examples count)

المطلوب:
1. تثبيت مكتبة: npm install bayes-opt

2. تعريف Parameter Space:
   - roleStyle: ['professional', 'casual', 'expert']
   - constraintCount: [0, 5]
   - exampleCount: [0, 3]
   - formatStyle: ['markdown', 'json', 'plain']

3. Objective Function: تعظيم score مع تقليل tokens

الوظيفة المطلوبة:
async function bayesianOptimize(
  template: PromptTemplate,
  testCases: TestCase[],
  iterations: number = 20
): Promise<OptimalParameters>

interface OptimalParameters {
  parameters: Record<string, any>;
  expectedScore: number;
  confidence: number;
}

الملف: `src/optimizer/bayesian.ts`
```

### DIRECTIVE-022: تنفيذ Bandits/MCTS للفضاءات الكبيرة [DONE]

```
المهمة: أنشئ optimizer يستخدم Multi-Armed Bandits أو MCTS

الاستخدام: عندما يكون عدد الـ mutations المحتملة كبير جداً

**Multi-Armed Bandits**:
- كل mutation type هو "arm"
- قيّم أداء كل arm
- اختر الـ arms الأفضل أداءً بشكل متكرر (exploitation)
- جرّب arms جديدة أحياناً (exploration)

الوظيفة المطلوبة:
function banditOptimize(
  prompt: string,
  availableMutations: MutationType[],
  budget: number  // عدد المحاولات المتاحة
): BanditResult

interface BanditResult {
  bestMutations: MutationType[];
  expectedRewards: number[];
  explorationRate: number;
}

**MCTS (Monte Carlo Tree Search)**:
- بناء شجرة من الـ mutations الممكنة
- استكشاف الفروع الواعدة بشكل أعمق
- موازنة exploration vs exploitation

الملفات:
- `src/optimizer/bandits.ts`
- `src/optimizer/mcts.ts`
```

### DIRECTIVE-023: إعداد نظام RL (PPO-like)

```
المهمة: أنشئ نظام Reinforcement Learning لتحسين سياسة التوليد

تحذير: هذه مهمة متقدمة جداً! تحتاج إلى:
1. Reward Model مدرب
2. Policy Network
3. Value Network
4. PPO Training Loop

الخطوات:
1. **أنشئ Policy Network**:
   - Input: embedding للـ prompt الأصلي
   - Output: distribution على الـ mutation actions

2. **أنشئ Value Network**:
   - Input: embedding للـ prompt
   - Output: تقدير للـ expected reward

3. **PPO Training**:
   - جمّع experiences (prompt, action, reward)
   - احسب advantages
   - حدّث Policy بحذر (clipped objective)

الملفات:
- `src/rl/policy.py` (استخدم Python + PyTorch)
- `src/rl/value.py`
- `src/rl/ppo_trainer.py`
- `src/rl/interface.ts` (TypeScript wrapper)

الموارد المطلوبة: GPU للتدريب

ملاحظة: هذا للمرحلة المتقدمة جداً (Phase 3)
```

### DIRECTIVE-024: بناء Hybrid Optimizer [DONE]

```
المهمة: ادمج عدة optimizers في نظام هجين ذكي

الاستراتيجية:
1. **المرحلة 1 (Exploration)**: استخدم Genetic Algorithm
   - ولّد population متنوعة (20 variations)
   - شغّل لـ 3-5 أجيال
   - احصل على أفضل 5

2. **المرحلة 2 (Refinement)**: استخدم Hill-Climbing
   - ابدأ من كل واحد من الـ 5 الأفضل
   - طبّق hill-climbing لـ 5 iterations
   - احصل على الأفضل من كل branch

3. **المرحلة 3 (Fine-tuning)**: استخدم Bayesian Optimization
   - خذ الأفضل من المرحلة 2
   - حسّن معلماته بدقة

الوظيفة المطلوبة:
async function hybridOptimize(
  prompt: string,
  config: HybridConfig
): Promise<HybridResult>

interface HybridConfig {
  explorationBudget: number;   // عدد evaluations للمرحلة 1
  refinementBudget: number;    // عدد evaluations للمرحلة 2
  finetuningBudget: number;    // عدد evaluations للمرحلة 3
}

الملف: `src/optimizer/hybrid.ts`
```

---

## 🧪 المرحلة 4: بناء Sandbox & Testing Environment

### DIRECTIVE-025: بناء Test Suite Executor [DONE]

```
المهمة: أنشئ نظام لتشغيل prompts على test cases متوازية

المكونات المطلوبة:
1. **Test Case Definition**:
interface TestCase {
  id: string;
  prompt: string;
  expectedOutput?: string;
  evaluationCriteria: EvaluationCriteria;
  metadata?: Record<string, any>;
}

2. **Parallel Executor**:
async function executeTestSuite(
  promptVariations: string[],
  testCases: TestCase[],
  maxConcurrency: number = 5
): Promise<TestResults>

3. **Results Aggregation**:
interface TestResults {
  variationId: string;
  results: TestCaseResult[];
  aggregateScore: number;
  passRate: number;
}

الميزات المطلوبة:
- تشغيل متوازي مع rate limiting
- retry logic للفشل المؤقت
- timeout handling
- progress reporting

الملف: `src/sandbox/testExecutor.ts`
```

### DIRECTIVE-026: إضافة Caching للنتائج [DONE]

```
المهمة: أضف نظام caching ذكي لتقليل API calls

أنواع الـ Cache المطلوبة:
1. **Prompt Cache**: لتخزين مخرجات prompts مطابقة
2. **Embedding Cache**: لتخزين embeddings محسوبة
3. **Evaluation Cache**: لتخزين scores محسوبة

الوظائف المطلوبة:
class PromptCache {
  async get(prompt: string, provider: string): Promise<string | null>
  async set(prompt: string, provider: string, output: string, ttl?: number)
  async invalidate(pattern: string)
  getStats(): CacheStats
}

استراتيجية الـ Cache:
- استخدم hash للـ prompt كمفتاح
- TTL: 7 أيام للنتائج
- LRU eviction عند امتلاء الذاكرة
- خيار persistent storage (Redis)

الملف: `src/cache/promptCache.ts`
التبعيات: node-cache أو ioredis
```

### DIRECTIVE-027: إعداد Reference Datasets

```
المهمة: أنشئ datasets مرجعية لأنواع مختلفة من البرومبتات

المطلوب:
1. **Code Generation Dataset** (20 examples):
   - مهام برمجية متنوعة
   - مخرجات مرجعية صحيحة
   - معايير تقييم

2. **Content Writing Dataset** (20 examples):
   - مواضيع مختلفة
   - أساليب مختلفة (formal, casual, technical)
   - معايير جودة

3. **Marketing Copy Dataset** (20 examples):
   - منتجات مختلفة
   - CTA متنوعة
   - tone variations

البنية:
const datasets = {
  code: CodeDataset[],
  content: ContentDataset[],
  marketing: MarketingDataset[]
}

الملفات:
- `src/datasets/code.json`
- `src/datasets/content.json`
- `src/datasets/marketing.json`
- `src/datasets/loader.ts`

المصادر: استخدم prompts حقيقية من مشاريع أو أنشئها يدوياً
```

### DIRECTIVE-028: بناء نظام Lineage Tracking

```
المهمة: تتبع سلسلة النسب لكل variation (من أين جاء، لماذا، النتائج)

البنية المطلوبة:
interface VariationLineage {
  id: string;
  parentId: string | null;
  originalPrompt: string;
  mutation: MutationType;
  mutationParams: Record<string, any>;
  timestamp: Date;
  score: number;
  cost: number;
  latency: number;
  feedback?: HumanFeedback;
  children: string[];  // IDs of variations derived from this one
}

الوظائف المطلوبة:
1. `trackVariation(variation: VariationLineage): void`
2. `getLineage(variationId: string): VariationLineage[]`
3. `visualizeLineage(variationId: string): LineageGraph`
4. `findBestPath(originalPrompt: string, targetScore: number): VariationLineage[]`

قاعدة البيانات:
CREATE TABLE variation_lineage (
  id VARCHAR PRIMARY KEY,
  parent_id VARCHAR REFERENCES variation_lineage(id),
  original_prompt TEXT,
  mutation VARCHAR,
  mutation_params JSONB,
  timestamp TIMESTAMP,
  score FLOAT,
  cost FLOAT,
  latency FLOAT,
  feedback JSONB
);

الملف: `src/lineage/tracker.ts`
```

---

## 👥 المرحلة 5: Human-in-the-Loop System

### DIRECTIVE-029: بناء Sample Selection للمراجعة البشرية

```
المهمة: أنشئ نظام ذكي لاختيار عينات للمراجعة البشرية

استراتيجيات الاختيار:
1. **Uncertainty Sampling**: اختر variations حيث النموذج غير متأكد
2. **Diversity Sampling**: اختر variations متنوعة
3. **Error Analysis**: اختر variations التي فشلت في tests
4. **Random Sampling**: عينة عشوائية للتحقق

الوظيفة المطلوبة:
function selectSamplesForReview(
  variations: ScoredSuggestion[],
  strategy: SamplingStrategy,
  count: number
): ScoredSuggestion[]

enum SamplingStrategy {
  UNCERTAINTY,
  DIVERSITY,
  ERROR_FOCUSED,
  RANDOM,
  MIXED
}

الملف: `src/humanLoop/sampleSelection.ts`
الهدف: 5-10% من الـ variations للمراجعة البشرية
```

### DIRECTIVE-030: تطوير واجهة المراجعة البشرية

```
المهمة: أنشئ UI بسيطة للموافقة/الرفض والتعديل اليدوي

المكونات المطلوبة:
1. **Review Queue Component**:
   - قائمة بالـ variations المطلوب مراجعتها
   - تصفية حسب الأولوية والفئة
   - عداد للتقدم

2. **Review Card Component**:
   - عرض Original vs Suggested
   - المقاييس (Score, Cost, etc.)
   - أزرار: ✅ Approve, ❌ Reject, ✏️ Edit
   - حقل ملاحظات

3. **Edit Modal**:
   - text editor لتعديل الـ variation
   - معاينة فورية للمقاييس
   - حفظ كـ "human-refined" variation

الملفات:
- `src/components/ReviewQueue.tsx`
- `src/components/ReviewCard.tsx`
- `src/components/EditModal.tsx`

الـ API Endpoints:
- GET /api/review/queue
- POST /api/review/approve
- POST /api/review/reject
- PUT /api/review/edit
```

---

## 🛡️ المرحلة 6: Governance & Safety

### DIRECTIVE-031: تطوير قيود الأمان

```
المهمة: أنشئ نظام فحص أمان قبل تطبيق التعديلات

الفحوصات المطلوبة:
1. **Prompt Injection Detection**: هل الـ variation يحتوي على injection؟
2. **Sensitive Data Check**: هل يطلب معلومات حساسة؟
3. **Harmful Content**: هل يولّد محتوى ضار؟
4. **Bias Detection**: هل يحتوي على تحيز واضح؟

الوظيفة المطلوبة:
async function checkSafety(variation: string): Promise<SafetyReport>

interface SafetyReport {
  isSafe: boolean;
  violations: SafetyViolation[];
  confidence: number;
  recommendations: string[];
}

interface SafetyViolation {
  type: 'injection' | 'sensitive_data' | 'harmful' | 'bias';
  severity: 'low' | 'medium' | 'high';
  description: string;
  location: string;  // موقع المشكلة في النص
}

الملف: `src/safety/checker.ts`
الأدوات: استخدم OpenAI Moderation API + custom rules
```

### DIRECTIVE-032: بناء نظام Rollback/Preview

```
المهمة: أضف نظام معاينة وتراجع قبل تطبيق التغييرات

الميزات المطلوبة:
1. **Preview Mode**:
   - تشغيل الـ variation على sample inputs
   - عرض المخرجات جنباً إلى جنب مع الأصلي
   - إحصائيات مقارنة

2. **Staging Environment**:
   - تطبيق الـ variation في بيئة staging
   - مراقبة الأداء لفترة
   - A/B testing مع المستخدمين

3. **Rollback System**:
   - حفظ snapshot من الـ prompt الحالي
   - زر "Revert to Previous" في أي وقت
   - version history كاملة

الوظائف المطلوبة:
1. `previewVariation(variation: string, sampleInputs: string[]): PreviewResult`
2. `deployToStaging(variation: string): StagingDeployment`
3. `rollback(snapshotId: string): void`
4. `getVersionHistory(promptId: string): Version[]`

الملفات:
- `src/deployment/preview.ts`
- `src/deployment/staging.ts`
- `src/deployment/rollback.ts`
```

---

## 📚 المرحلة 7: البيانات والتعلّم (Training Infrastructure)

### DIRECTIVE-033: إعداد بيانات التدريب

```
المهمة: أنشئ pipeline لجمع وإعداد بيانات التدريب

البنية المطلوبة:
interface TrainingExample {
  id: string;
  originalPrompt: string;
  modifiedPrompt: string;
  context?: string;
  outputs: {
    original: string;
    modified: string;
  };
  humanScore: number;  // 1-5
  feedback?: string;
  metadata: {
    category: PromptCategory;
    mutationType: string;
    timestamp: Date;
    userId?: string;
  };
}

الوظائف المطلوبة:
1. `collectTrainingData(): AsyncGenerator<TrainingExample>`
   - جمع من human feedback
   - جمع من A/B testing results
   - جمع من lineage tracking

2. `cleanTrainingData(data: TrainingExample[]): TrainingExample[]`
   - إزالة duplicates
   - إزالة بيانات منخفضة الجودة
   - normalization

3. `splitDataset(data: TrainingExample[], trainRatio: number = 0.8)`
   - تقسيم إلى train/val/test
   - stratified sampling حسب الفئة

4. `exportForTraining(data: TrainingExample[], format: 'json' | 'csv' | 'parquet')`

الملفات:
- `src/training/dataCollection.ts`
- `src/training/dataPrep.ts`
```

### DIRECTIVE-034: بناء Reward Model

```
المهمة: درّب نموذج صغير للتنبؤ بجودة الـ variations

الخطوات:
1. **Data Preparation**:
   - جهّز dataset من TrainingExamples
   - features: [prompt_embedding, variation_embedding, metadata]
   - target: humanScore (normalized 0-1)

2. **Model Architecture**:
   - Transformer-based أو BERT-like
   - أو نموذج أبسط (XGBoost/Random Forest) للبداية

3. **Training**:
   - loss: MSE أو Huber loss
   - optimizer: AdamW
   - validation على hold-out set
   - early stopping

4. **Evaluation**:
   - MAE, RMSE على test set
   - correlation مع human scores
   - calibration check

الملفات (Python):
- `models/reward_model.py`
- `models/train_reward.py`
- `models/evaluate_reward.py`

الملف (TypeScript integration):
- `src/models/rewardModel.ts` - wrapper للاستدعاء

الموارد: GPU مستحسن للتدريب
```

### DIRECTIVE-035: تنفيذ RLAIF (RL from AI Feedback)

```
المهمة: قلل الاعتماد على البشر باستخدام AI للتقييم

الاستراتيجية:
1. **Bootstrap من بيانات بشرية**:
   - درّب Reward Model على human feedback أولي
   - استخدم النموذج لتوليد "AI feedback"

2. **Self-Play Loop**:
   - ولّد variations
   - قيّمها بالـ Reward Model
   - حسّن الـ Policy بناءً على التقييم
   - كرر

3. **Human-in-the-Loop Validation**:
   - راجع عينات دورياً مع بشر
   - صحّح أخطاء الـ Reward Model
   - أعد تدريب النموذج

الوظيفة المطلوبة:
async function rlaifTraining(
  initialPolicy: Policy,
  rewardModel: RewardModel,
  iterations: number
): Promise<ImprovedPolicy>

الملف: `src/training/rlaif.ts`
ملاحظة: هذا للمرحلة المتقدمة
```

---

## ⚙️ المرحلة 8: Production Infrastructure

### DIRECTIVE-036: تنفيذ Batching للطلبات

```
المهمة: أضف batching ذكي لتقليل تكلفة API calls

الاستراتيجية:
1. **Request Queuing**: اجمع الطلبات الواردة في queue
2. **Batch Formation**: اجمع طلبات متشابهة معاً
3. **Batch Processing**: شغّل الـ batch مرة واحدة
4. **Result Distribution**: وزّع النتائج على الطلبات الأصلية

الوظيفة المطلوبة:
class BatchProcessor {
  constructor(config: BatchConfig)

  async process(request: ProcessRequest): Promise<ProcessResult>

  // Internal
  private queue: Request[]
  private processBatch(batch: Request[]): Promise<Result[]>
  private formBatches(): Request[][]
}

interface BatchConfig {
  maxBatchSize: number;
  maxWaitTime: number;  // milliseconds
  similarityThreshold: number;  // لتجميع prompts متشابهة
}

الملف: `src/processing/batchProcessor.ts`
الفائدة: تقليل تكلفة API بنسبة 30-50%
```

### DIRECTIVE-037: إضافة Surrogate Models

```
المهمة: استخدم نماذج صغيرة/سريعة للتقييم الأولي

المفهوم:
- استخدم نموذج كبير (GPT-4) للمخرجات النهائية فقط
- استخدم نماذج أصغر (GPT-3.5, Llama) للتقييم والاستكشاف
- هذا يقلل التكلفة بشكل كبير

الوظيفة المطلوبة:
class SurrogateOrchestrator {
  async evaluate(
    prompt: string,
    mode: 'exploration' | 'exploitation' | 'final'
  ): Promise<EvaluationResult>
}

استراتيجية الاختيار:
- exploration: استخدم أرخص نموذج (Groq/Llama)
- exploitation: استخدم نموذج متوسط (GPT-3.5)
- final: استخدم أفضل نموذج (GPT-4/Claude)

الملف: `src/models/surrogateOrchestrator.ts`
التوفير المتوقع: 60-80% من التكلفة
```

### DIRECTIVE-038: معالجة Prompt Overfitting

```
المهمة: تأكد من أن الـ prompts المُحسّنة تعمل على مدخلات متنوعة

الاستراتيجيات:
1. **Diverse Test Sets**: اختبر على examples متنوعة
2. **Cross-Validation**: K-fold validation للـ prompts
3. **Held-out Validation**: احتفظ بـ test set منفصل
4. **Regularization**: عاقب التعقيد الزائد في الـ prompts

الوظيفة المطلوبة:
async function detectOverfitting(
  prompt: string,
  trainResults: TestResults,
  valResults: TestResults
): Promise<OverfittingReport>

interface OverfittingReport {
  isOverfit: boolean;
  trainScore: number;
  valScore: number;
  gap: number;  // الفرق بين train و val
  recommendation: string;
}

القاعدة:
- إذا كان (trainScore - valScore) > threshold → overfitting
- الحل: simplify prompt, add regularization, get more data

الملف: `src/evaluation/overfittingDetector.ts`
```

---

## 📊 المرحلة 9: KPIs & Analytics

### DIRECTIVE-039: نظام قياس تحسن Score على Benchmark

```
المهمة: أنشئ نظام لقياس التحسن على benchmark suite

المطلوب:
1. **Benchmark Suite**: مجموعة ثابتة من test cases
2. **Baseline Scores**: النقاط الأساسية قبل التحسين
3. **Tracking System**: تتبع النقاط عبر الوقت
4. **Statistical Testing**: اختبار دلالة التحسن

الوظائف المطلوبة:
1. `runBenchmark(prompts: string[]): BenchmarkResults`
2. `compareWithBaseline(current: BenchmarkResults, baseline: BenchmarkResults): Comparison`
3. `trackProgress(results: BenchmarkResults): void`
4. `generateReport(): BenchmarkReport`

interface BenchmarkReport {
  avgScoreImprovement: number;  // %
  significanceLevel: number;    // p-value
  bestImprovement: TestCase;
  worstImprovement: TestCase;
  trends: TimeSeries;
}

الملف: `src/analytics/benchmark.ts`
```

### DIRECTIVE-040: قياس انخفاض avg tokens/call

```
المهمة: تتبع متوسط استهلاك التوكنات وتكلفته

المطلوب:
1. **Token Tracking**: سجّل tokens لكل API call
2. **Aggregation**: احسب المتوسطات حسب الفترة
3. **Cost Calculation**: حوّل إلى تكلفة مالية
4. **Visualization**: رسم بياني للاتجاه

الوظائف المطلوبة:
class TokenAnalytics {
  logTokenUsage(call: APICall): void
  getAverageTokens(timeRange: TimeRange): number
  getCostSavings(baseline: number): CostSavings
  generateTokenReport(): TokenReport
}

interface TokenReport {
  avgTokensPerCall: number;
  totalTokens: number;
  totalCost: number;
  reduction: number;  // % مقارنة بالـ baseline
  projectedMonthlySavings: number;
}

الملف: `src/analytics/tokenAnalytics.ts`
```

### DIRECTIVE-041: قياس رضا المستخدم (NPS/Accept Rate)

```
المهمة: أنشئ نظام لقياس رضا المستخدمين

المقاييس المطلوبة:
1. **Accept Rate**: نسبة الاقتراحات المقبولة
2. **NPS (Net Promoter Score)**: "هل ستوصي بهذه الأداة؟"
3. **User Satisfaction**: تقييم 1-5 نجوم
4. **Feature Usage**: أي mutations الأكثر استخداماً

الوظائف المطلوبة:
class UserSatisfactionTracker {
  logAcceptance(suggestionId: string, accepted: boolean): void
  logNPSScore(userId: string, score: number): void
  logSatisfactionRating(sessionId: string, rating: number): void

  getAcceptRate(timeRange: TimeRange): number
  getNPS(timeRange: TimeRange): number
  getAverageSatisfaction(timeRange: TimeRange): number

  generateSatisfactionReport(): SatisfactionReport
}

الـ UI المطلوبة:
- استبيان NPS بعد كل جلسة (أحياناً)
- تقييم سريع بعد قبول/رفض اقتراح
- "Was this helpful?" بعد كل نتيجة

الملف: `src/analytics/userSatisfaction.ts`
```

### DIRECTIVE-042: قياس Adoption Rate للاقتراحات

```
المهمة: تتبع معدل تبني الاقتراحات المختلفة

المقاييس:
1. **Overall Adoption**: نسبة المستخدمين الذين يستخدمون الميزة
2. **Mutation Adoption**: أي mutation types الأكثر قبولاً
3. **Category Adoption**: أداء كل category
4. **Time to Adoption**: كم يستغرق المستخدم لقبول اقتراح

الوظائف المطلوبة:
class AdoptionTracker {
  trackSuggestionShown(suggestionId: string, metadata: SuggestionMetadata): void
  trackSuggestionAccepted(suggestionId: string, timeToAccept: number): void

  getAdoptionRate(dimension: 'overall' | 'mutation' | 'category'): AdoptionMetrics
  getTimeToAdoption(): number

  generateAdoptionReport(): AdoptionReport
}

interface AdoptionReport {
  overallRate: number;
  byMutation: Map<MutationType, number>;
  byCategory: Map<PromptCategory, number>;
  avgTimeToAdopt: number;
  trends: TimeSeries;
}

الملف: `src/analytics/adoption.ts`
```

---

## 🛠️ المرحلة 10: التقنيات والأدوات (Tech Stack)

### DIRECTIVE-043: إعداد LangChain Pipelines

```
المهمة: أنشئ orchestration pipelines باستخدام LangChain

التثبيت:
npm install langchain @langchain/core @langchain/openai

المطلوب:
1. **Refinement Pipeline**:
   Input: original prompt
   Steps:
   - Classification
   - Mutation generation
   - Evaluation
   - Ranking
   Output: top 3 suggestions

2. **Multi-Step Pipeline**:
   - تقسيم المهمة
   - تشغيل الخطوات
   - تجميع النتائج

3. **RAG Pipeline** (للـ factuality):
   - Retrieval من knowledge base
   - Verification
   - Scoring

الملفات:
- `src/pipelines/refinementPipeline.ts`
- `src/pipelines/multiStepPipeline.ts`
- `src/pipelines/ragPipeline.ts`

مثال:
```typescript
import { RunnableSequence } from "@langchain/core/runnables";

const refinementPipeline = RunnableSequence.from([
  classifyPrompt,
  generateMutations,
  evaluateInParallel,
  rankAndFilter
]);

const result = await refinementPipeline.invoke({ prompt: "..." });
```

```

### DIRECTIVE-044: إعداد Vector Database (Pinecone/Weaviate)
```

المهمة: أنشئ vector database لتخزين prompts و embeddings

الاختيار: Pinecone (سهل) أو Weaviate (open-source)

**Option 1: Pinecone**

```typescript
import { PineconeClient } from "@pinecone-database/pinecone";

1. إنشاء index للـ prompts
2. إنشاء index للـ test cases
3. إنشاء index للـ knowledge base (RAG)
```

**Option 2: Weaviate**

```typescript
import weaviate from 'weaviate-ts-client';

1. نفس الـ indices
2. ميزة: self-hosted، أرخص
```

الـ Schema المطلوب:

- Prompts Collection: {prompt, embedding, category, metadata}
- TestCases Collection: {input, expected, embedding}
- Knowledge Collection: {text, embedding, source, timestamp}

الوظائف المطلوبة:

1. `indexPrompt(prompt: string, metadata: any): Promise<string>`
2. `searchSimilar(prompt: string, k: number): Promise<SearchResult[]>`
3. `retrieveContext(query: string): Promise<string[]>`

الملف: `src/vectorstore/client.ts`

```

### DIRECTIVE-045: دمج GROQ Models
```

المهمة: أضف دعماً لـ GROQ كـ provider بديل (سريع ورخيص)

التثبيت:
npm install groq-sdk

الاستخدام:

- exploration phase: استخدم Groq (أسرع وأرخص)
- final evaluation: استخدم OpenAI/Anthropic (أفضل جودة)

الوظائف المطلوبة:
class GroqProvider implements LLMProvider {
  async complete(prompt: string, config: CompletionConfig): Promise<string>
  async embed(text: string): Promise<number[]>
  estimateCost(tokens: number): number
  estimateLatency(tokens: number): number
}

الملف: `src/providers/groq.ts`

الإعدادات:

- Model: llama-3.1-70b (للجودة) أو llama-3.1-8b (للسرعة)
- Temperature: 0.7
- Max tokens: حسب الحاجة

```

### DIRECTIVE-046: إعداد Postgres + Prisma
```

المهمة: أنشئ قاعدة بيانات لتخزين البيانات الدائمة

التثبيت:
npm install prisma @prisma/client
npx prisma init

الـ Schema المطلوب (schema.prisma):

```prisma
model Prompt {
  id          String   @id @default(uuid())
  text        String
  category    String
  userId      String?
  createdAt   DateTime @default(now())
  variations  Variation[]
  feedback    Feedback[]
}

model Variation {
  id          String   @id @default(uuid())
  promptId    String
  prompt      Prompt   @relation(fields: [promptId], references: [id])
  text        String
  mutation    String
  score       Float
  tokenCount  Int
  cost        Float
  createdAt   DateTime @default(now())
  lineage     Lineage?
}

model Feedback {
  id          String   @id @default(uuid())
  promptId    String
  prompt      Prompt   @relation(fields: [promptId], references: [id])
  variationId String?
  score       Int
  comment     String?
  userId      String
  createdAt   DateTime @default(now())
}

model Lineage {
  id            String    @id @default(uuid())
  variationId   String    @unique
  variation     Variation @relation(fields: [variationId], references: [id])
  parentId      String?
  mutationParams Json?
  score         Float
  createdAt     DateTime  @default(now())
}

model TestCase {
  id          String   @id @default(uuid())
  prompt      String
  expected    String?
  category    String
  metadata    Json?
  createdAt   DateTime @default(now())
}
```

الملفات:

- `prisma/schema.prisma`
- `src/db/client.ts`

```

### DIRECTIVE-047: إعداد Object Storage
```

المهمة: أضف تخزين للملفات الكبيرة (datasets, models, logs)

الخيارات:

- AWS S3
- Google Cloud Storage
- MinIO (self-hosted)

الاستخدام:

1. تخزين training datasets
2. تخزين model checkpoints
3. تخزين logs طويلة
4. تخزين exported reports

الوظائف المطلوبة:
class ObjectStore {
  async upload(key: string, data: Buffer | Stream): Promise<string>
  async download(key: string): Promise<Buffer>
  async delete(key: string): Promise<void>
  async list(prefix: string): Promise<string[]>
  async getSignedUrl(key: string, expiresIn: number): Promise<string>
}

الملف: `src/storage/objectStore.ts`

```

### DIRECTIVE-048: إعداد Kubernetes + Autoscaling
```

المهمة: أنشئ deployment على Kubernetes مع autoscaling

الملفات المطلوبة:

1. **Dockerfile**:

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

2. **k8s/deployment.yaml**:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prompt-refiner
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prompt-refiner
  template:
    metadata:
      labels:
        app: prompt-refiner
    spec:
      containers:
      - name: app
        image: prompt-refiner:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
```

3. **k8s/hpa.yaml** (Horizontal Pod Autoscaler):

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: prompt-refiner-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: prompt-refiner
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

الملفات: `k8s/deployment.yaml`, `k8s/service.yaml`, `k8s/hpa.yaml`

```

### DIRECTIVE-049: إعداد عُقد GPU للـ RL/Fine-tune
```

المهمة: أضف GPU nodes لتدريب النماذج

الخيارات:

1. **Cloud GPU**: AWS p3/p4, GCP A100, Azure NC-series
2. **GPU-as-a-Service**: Lambda Labs, Paperspace, RunPod

الإعداد المطلوب:

1. **GPU Node Pool** في Kubernetes
2. **Job Scheduler** للتدريب
3. **Model Registry** لحفظ النماذج المدربة

الملفات:

- `k8s/gpu-nodepool.yaml`
- `training/train-job.yaml`

مثال Training Job:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: reward-model-training
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: pytorch/pytorch:2.0-cuda11.8
        command: ["python", "train_reward.py"]
        resources:
          limits:
            nvidia.com/gpu: 1
      restartPolicy: Never
```

```

### DIRECTIVE-050: إعداد Prometheus + Grafana
```

المهمة: أنشئ نظام monitoring شامل

المكونات:

1. **Prometheus**: جمع المقاييس
2. **Grafana**: عرض الـ dashboards
3. **Alert Manager**: إرسال تنبيهات

المقاييس المطلوب تتبعها:

- Request rate (requests/second)
- Error rate (%)
- Response time (p50, p95, p99)
- Token usage (tokens/hour)
- Cost ($/hour)
- Cache hit rate (%)
- Model latency (ms)
- Queue depth

الملفات:

- `monitoring/prometheus.yml`
- `monitoring/grafana-dashboard.json`
- `src/metrics/collector.ts`

مثال Metrics Collection:

```typescript
import client from 'prom-client';

const requestCounter = new client.Counter({
  name: 'refiner_requests_total',
  help: 'Total number of refinement requests'
});

const tokenHistogram = new client.Histogram({
  name: 'refiner_tokens_used',
  help: 'Tokens used per request',
  buckets: [10, 50, 100, 500, 1000, 5000]
});
```

```

### DIRECTIVE-051: إعداد Feature Flags لـ A/B Testing
```

المهمة: أضف نظام feature flags للتجارب

الخيارات:

- LaunchDarkly (مدفوع، قوي)
- Unleash (open-source)
- PostHog (analytics + flags)

الاستخدام:

1. **A/B Testing للـ Mutations**: أي mutation أفضل؟
2. **Gradual Rollout**: تدريجياً نشر ميزات جديدة
3. **Emergency Kill Switch**: إيقاف ميزة بسرعة

الوظائف المطلوبة:
class FeatureFlags {
  async isEnabled(flagName: string, userId?: string): Promise<boolean>
  async getVariant(experiment: string, userId: string): Promise<string>
  async track(event: string, properties: any): Promise<void>
}

الأعلام المطلوبة:

- `use_genetic_optimizer`: تفعيل genetic algorithm
- `enable_rl_policy`: استخدام RL policy
- `show_advanced_metrics`: عرض مقاييس متقدمة
- `enable_human_review`: تفعيل المراجعة البشرية

الملف: `src/features/flags.ts`

```

---

## 🚀 المرحلة 11: Population Search + Sandbox (المرحلة 2)

### DIRECTIVE-052: تنفيذ Population Search
```

المهمة: أنشئ نظام بحث قائم على السكان

الخطوات:

1. **Initialize**: ولّد 20-50 variation عشوائية
2. **Evaluate**: قيّم كل واحد على test suite
3. **Select**: اختر أفضل 50%
4. **Evolve**: طبّق mutations على المختارين
5. **Repeat**: كرر لعدة أجيال

الوظيفة المطلوبة:
async function populationSearch(
  initialPrompt: string,
  testSuite: TestCase[],
  config: PopulationConfig
): Promise<PopulationResult>

interface PopulationConfig {
  populationSize: number;      // 20-50
  generations: number;         // 5-10
  selectionRate: number;       // 0.5
  mutationProbability: number; // 0.3
  crossoverProbability: number;// 0.7
}

الملف: `src/search/populationSearch.ts`
الهدف: إيجاد variations متنوعة وعالية الجودة

```

### DIRECTIVE-053: تطوير Sandbox Run على Test Suite
```

المهمة: شغّل كل variation على مجموعة اختبار كاملة

الميزات:

1. **Parallel Execution**: شغّل عدة tests بالتوازي
2. **Timeout Handling**: أوقف tests طويلة
3. **Error Isolation**: لا تدع خطأ واحد يوقف الكل
4. **Result Aggregation**: اجمع النتائج وقارنها

الوظيفة المطلوبة:
async function sandboxRun(
  variation: string,
  testSuite: TestCase[],
  config: SandboxConfig
): Promise<SandboxResult>

interface SandboxConfig {
  maxConcurrency: number;
  timeoutPerTest: number;    // milliseconds
  retryOnFailure: boolean;
  collectOutputs: boolean;
}

interface SandboxResult {
  variationId: string;
  passedTests: number;
  failedTests: number;
  avgScore: number;
  outputs: Map<string, string>;  // testId -> output
  errors: Map<string, Error>;
}

الملف: `src/sandbox/sandboxRunner.ts`

```

### DIRECTIVE-054: جمع Human Feedback وبناء Reward Dataset
```

المهمة: أنشئ pipeline لجمع feedback وتحويله لـ training data

الخطوات:

1. **Collection**: اجمع feedback من UI
2. **Validation**: تحقق من جودة البيانات
3. **Augmentation**: أضف features (embeddings, metadata)
4. **Storage**: خزّن في database
5. **Export**: صدّر للتدريب

الوظائف المطلوبة:

1. `collectFeedback(variationId: string, feedback: Feedback): Promise<void>`
2. `validateFeedback(feedback: Feedback): boolean`
3. `buildRewardDataset(filters?: DatasetFilters): Promise<RewardDataset>`
4. `exportDataset(dataset: RewardDataset, format: ExportFormat): Promise<string>`

interface RewardDataset {
  examples: RewardExample[];
  statistics: DatasetStats;
  metadata: {
    created: Date;
    version: string;
    size: number;
  };
}

interface RewardExample {
  promptEmbedding: number[];
  variationEmbedding: number[];
  features: number[];  // [tokenCount, similarity, etc.]
  label: number;       // normalized human score
  weight: number;      // confidence/importance
}

الملف: `src/training/rewardDatasetBuilder.ts`

```

### DIRECTIVE-055: إضافة Hallucination Checker
```

المهمة: دمج hallucination detection في pipeline الرئيسي

الطرق المستخدمة:

1. **Self-Consistency**: شغّل مرتين واقارن
2. **Retrieval Check**: تحقق من الحقائق عبر RAG
3. **Confidence Analysis**: استخدم logprobs

التكامل المطلوب:

- أضف hallucination score في ScoredSuggestion
- أضف تحذير في UI إذا كان الـ score عالي
- رفض variations مع hallucination عالية تلقائياً

الوظيفة المحدّثة:
async function evaluateSuggestions(
  originalPrompt: string,
  variations: PromptVariation[],
  checkHallucination: boolean = true
): Promise<ScoredSuggestion[]>

// في ScoredSuggestion:
interface ScoredSuggestion {
  // ... existing fields
  hallucinationScore?: number;
  hallucinationWarning?: string;
}

الملف: `src/evaluator.ts` (تحديث)

```

### DIRECTIVE-056: دمج RAG للـ Factuality
```

المهمة: استخدم RAG للتحقق من صحة الحقائق في المخرجات

الخطوات:

1. **Setup Knowledge Base**:
   - جمّع مصادر موثوقة (Wikipedia, docs, etc.)
   - حوّلها لـ embeddings
   - خزّنها في vector DB

2. **Retrieval Function**:
   async function retrieveRelevantFacts(claim: string): Promise<Fact[]>

3. **Verification Function**:
   async function verifyAgainstFacts(
     claim: string,
     facts: Fact[]
   ): Promise<VerificationResult>

4. **Integration**:
   - أضف factuality score في التقييم
   - عرض المصادر الداعمة/المتناقضة في UI

الملفات:

- `src/rag/knowledgeBase.ts`
- `src/rag/factVerifier.ts`
- `src/evaluator.ts` (تحديث)

```

---

## 🤖 المرحلة 12: Reward Model + RL (المرحلة 3)

### DIRECTIVE-057: بناء Reward Model (Fine-tune/Supervised)
```

المهمة: درّب نموذج للتنبؤ بجودة الـ variations

الخطوات (Python + PyTorch):

1. **Prepare Data**:

```python
# data format
{
  "prompt": "...",
  "variation": "...",
  "score": 0.85,
  "metadata": {...}
}
```

2. **Model Architecture**:

```python
import torch.nn as nn
from transformers import AutoModel

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        self.regressor = nn.Sequential(
            nn.Linear(768 * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, prompt_emb, variation_emb):
        combined = torch.cat([prompt_emb, variation_emb], dim=1)
        return self.regressor(combined)
```

3. **Training**:

```python
# train.py
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

for epoch in range(epochs):
    for batch in dataloader:
        loss = criterion(model(batch.prompt, batch.variation), batch.score)
        loss.backward()
        optimizer.step()
```

4. **TypeScript Integration**:

```typescript
// src/models/rewardModel.ts
class RewardModel {
  async predict(prompt: string, variation: string): Promise<number>
  async batchPredict(pairs: [string, string][]): Promise<number[]>
}
```

الملفات:

- `models/reward_model.py`
- `models/train.py`
- `src/models/rewardModel.ts`

```

### DIRECTIVE-058: تطبيق PPO لتحسين سياسة التوليد
```

المهمة: استخدم PPO لتدريب policy network

تحذير: مهمة متقدمة جداً!

المكونات (Python):

1. **Policy Network**: يختار mutation action
2. **Value Network**: يقدّر expected reward
3. **PPO Trainer**: يحدّث Networks

```python
# policy.py
class MutationPolicy(nn.Module):
    def __init__(self, action_space_size):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("...")
        self.actor = nn.Linear(768, action_space_size)

    def forward(self, prompt_emb):
        logits = self.actor(prompt_emb)
        return F.softmax(logits, dim=-1)

# ppo_trainer.py
class PPOTrainer:
    def train_step(self, experiences):
        # 1. Compute advantages
        advantages = self.compute_advantages(experiences)

        # 2. Update policy with clipped objective
        ratio = new_policy / old_policy
        clipped = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        loss = -torch.min(ratio * advantages, clipped * advantages).mean()

        # 3. Update value network
        value_loss = F.mse_loss(predicted_values, returns)
```

الاستخدام:

1. جمّع experiences (prompt, action, reward)
2. احسب advantages
3. حدّث policy
4. كرر

الملفات:

- `models/policy.py`
- `models/value.py`
- `models/ppo_trainer.py`

```

### DIRECTIVE-059: تنفيذ Multi-objective Optimization (Pareto)
```

المهمة: حسّن لعدة أهداف في نفس الوقت (cost vs accuracy vs latency)

المفهوم:

- لا يوجد حل واحد أفضل
- ابحث عن Pareto Front (حلول غير مسيطر عليها)
- دع المستخدم يختار من الـ Pareto set

الخوارزمية:

1. ولّد مجموعة من الـ variations
2. قيّم كل واحد على جميع الأهداف
3. احسب Pareto Front
4. عرض الخيارات للمستخدم

الوظيفة المطلوبة:
function paretoOptimize(
  variations: ScoredSuggestion[],
  objectives: Objective[]
): ParetoFront

interface Objective {
  name: string;
  getValue: (s: ScoredSuggestion) => number;
  minimize: boolean;  // true = lower is better
}

interface ParetoFront {
  solutions: ScoredSuggestion[];
  dominatedCount: number;
  paretoCount: number;
}

// تحديد dominance
function dominates(a: ScoredSuggestion, b: ScoredSuggestion): boolean {
  // a dominates b إذا كان أفضل في جميع الأهداف
}

الملف: `src/optimizer/pareto.ts`

```

### DIRECTIVE-060: بناء آليات A/B Testing
```

المهمة: أنشئ نظام A/B testing لتجريب variations

الميزات:

1. **Experiment Definition**:
   - Control group (prompt الأصلي)
   - Treatment groups (variations مختلفة)
   - Traffic split (50/50, 70/30, etc.)

2. **Random Assignment**: وزّع المستخدمين عشوائياً

3. **Metrics Collection**: اجمع مقاييس لكل group

4. **Statistical Analysis**: احسب significance

الوظائف المطلوبة:
class ABTest {
  constructor(config: ABTestConfig)

  assign(userId: string): string  // returns variant
  trackMetric(userId: string, metric: string, value: number)
  getResults(): ABTestResults

  // Statistical tests
  calculateSignificance(): number  // p-value
  getConfidenceInterval(metric: string): [number, number]
}

interface ABTestResults {
  control: GroupMetrics;
  treatments: Map<string, GroupMetrics>;
  winner?: string;
  significance: number;
  recommendation: string;
}

الملف: `src/experiments/abTesting.ts`

```

### DIRECTIVE-061: تطوير Canary Releases
```

المهمة: أضف نظام canary deployment للـ variations الجديدة

الاستراتيجية:

1. **Deploy to 5%** من المستخدمين
2. **Monitor** المقاييس لمدة ساعة
3. **Compare** مع baseline
4. **Decision**:
   - إذا جيد → زد إلى 25%
   - إذا ممتاز → زد إلى 100%
   - إذا سيء → rollback فوراً

الوظائف المطلوبة:
class CanaryDeployment {
  async deploy(variation: string, percentage: number): Promise<DeploymentId>
  async monitor(deploymentId: string): Promise<HealthMetrics>
  async scale(deploymentId: string, newPercentage: number): Promise<void>
  async rollback(deploymentId: string): Promise<void>

  // Auto decision
  async autoScale(deploymentId: string, criteria: ScalingCriteria): Promise<void>
}

interface ScalingCriteria {
  errorRateThreshold: number;
  latencyThreshold: number;
  satisfactionThreshold: number;
  minObservations: number;
}

الملف: `src/deployment/canary.ts`

```

### DIRECTIVE-062: تنفيذ Auto-deploy
```

المهمة: أضف deployment تلقائي عند نجاح الاختبارات

الشروط للـ Auto-deploy:

1. ✅ جميع unit tests تمر
2. ✅ Safety checks تمر
3. ✅ Canary deployment ناجح
4. ✅ A/B test يظهر تحسن ذو دلالة
5. ✅ Human approval (optional, configurable)

الوظيفة المطلوبة:
class AutoDeployer {
  async evaluateForDeploy(variation: string): Promise<DeployDecision>

  async deploy(variation: string, options: DeployOptions): Promise<Deployment>

  // Monitoring
  async monitorDeployment(deploymentId: string): Promise<void>
}

interface DeployDecision {
  shouldDeploy: boolean;
  confidence: number;
  checks: CheckResult[];
  recommendation: string;
}

workflow:

1. Variation created
2. Run tests
3. Canary deploy (5%)
4. Monitor (1 hour)
5. A/B test (24 hours)
6. Auto-decision
7. Full deploy or rollback

الملف: `src/deployment/autoDeployer.ts`

```

---

## 🏢 المرحلة 13: منتج مؤسسي (المرحلة 4)

### DIRECTIVE-063: تطوير Continuous Learning
```

المهمة: أنشئ نظام تعلم مستمر من الإنتاج

المكونات:

1. **Data Collection Pipeline**:
   - اجمع prompts + outputs من الإنتاج
   - اجمع user feedback تلقائياً
   - اجمع performance metrics

2. **Model Retraining**:
   - جدولة إعادة تدريب أسبوعية/شهرية
   - استخدم بيانات جديدة
   - قارن النموذج الجديد بالقديم
   - deploy إذا كان أفضل

3. **Feedback Loop**:
   - Model predictions → User interactions → Feedback → Training data → Improved model

الوظائف المطلوبة:
class ContinuousLearning {
  async collectProductionData(timeRange: TimeRange): Promise<Dataset>
  async triggerRetraining(dataset: Dataset): Promise<TrainingJob>
  async evaluateNewModel(modelId: string): Promise<EvaluationReport>
  async promoteModel(modelId: string): Promise<void>
}

الجدولة:

- Weekly: جمع بيانات جديدة
- Monthly: إعادة تدريب
- On-demand: عند الحاجة

الملف: `src/learning/continuousLearning.ts`

```

### DIRECTIVE-064: بناء Personalization لكل User/Org
```

المهمة: خصّص التوصيات حسب المستخدم/المنظمة

الميزات:

1. **User Preferences**:
   - Mutation types المفضلة
   - Balance weights مخصصة (cost vs quality)
   - Prompt categories الشائعة

2. **Learning from History**:
   - تتبع ما يقبله/يرفضه المستخدم
   - تعلّم preferences تلقائياً
   - حسّن التوصيات بمرور الوقت

3. **Org-level Settings**:
   - Brand voice guidelines
   - Technical constraints
   - Budget limits

الوظائف المطلوبة:
class PersonalizationEngine {
  async getUserProfile(userId: string): Promise<UserProfile>
  async updateProfile(userId: string, interaction: UserInteraction): Promise<void>
  async personalizeRecommendations(
    variations: ScoredSuggestion[],
    userId: string
  ): Promise<ScoredSuggestion[]>
}

interface UserProfile {
  userId: string;
  preferences: {
    favoredMutations: MutationType[];
    balanceWeights: BalanceMetrics;
    stylePreferences: StyleGuide;
  };
  history: {
    acceptedVariations: string[];
    rejectedVariations: string[];
    avgAcceptanceTime: number;
  };
  orgSettings?: OrgSettings;
}

الملف: `src/personalization/engine.ts`

```

### DIRECTIVE-065: تطوير Explainability
```

المهمة: اشرح لماذا اقترح النظام variation معينة

الأسئلة للإجابة عليها:

1. "لماذا هذا الاقتراح؟"
2. "ما أثر كل تغيير؟"
3. "كيف يحسّن الجودة/التكلفة/السرعة؟"

الوظائف المطلوبة:
function explainSuggestion(suggestion: ScoredSuggestion): Explanation

interface Explanation {
  summary: string;  // "This variation reduces cost by 30% while maintaining quality"

  changes: Change[];  // قائمة بالتغييرات المحددة

  impact: {
    quality: ImpactAnalysis;
    cost: ImpactAnalysis;
    latency: ImpactAnalysis;
  };

  reasoning: string;  // شرح تفصيلي

  tradeoffs: string[];  // المقايضات
}

interface Change {
  type: 'addition' | 'removal' | 'modification';
  text: string;
  reason: string;
  impact: string;
}

interface ImpactAnalysis {
  direction: 'improved' | 'degraded' | 'neutral';
  magnitude: number;  // percentage
  confidence: number;
  explanation: string;
}

UI Component:

- زر "Why this suggestion?" لكل variation
- Modal يعرض Explanation بشكل واضح
- Diff view للتغييرات

الملف: `src/explainability/explainer.ts`

```

### DIRECTIVE-066: بناء Marketplace للـ Policies والقوالب
```

المهمة: أنشئ marketplace لمشاركة وتداول الـ policies

الميزات:

1. **Policy Library**:
   - Mutation policies
   - Evaluation policies
   - Optimization strategies
   - Prompt templates

2. **Sharing & Discovery**:
   - نشر policy
   - بحث واكتشاف
   - تقييمات ومراجعات
   - تحميل واستخدام

3. **Versioning & Updates**:
   - version control للـ policies
   - تحديثات تلقائية (optional)
   - changelog

البنية المطلوبة:
interface Policy {
  id: string;
  name: string;
  description: string;
  author: string;
  version: string;
  category: PolicyCategory;
  config: any;  // Policy-specific config
  tags: string[];
  downloads: number;
  rating: number;
  reviews: Review[];
}

class Marketplace {
  async publishPolicy(policy: Policy): Promise<string>
  async searchPolicies(query: string, filters: PolicyFilters): Promise<Policy[]>
  async downloadPolicy(policyId: string): Promise<Policy>
  async ratePolicy(policyId: string, rating: number, review?: string): Promise<void>
  async updatePolicy(policyId: string, updates: Partial<Policy>): Promise<void>
}

الملفات:

- `src/marketplace/marketplace.ts`
- `src/marketplace/policyManager.ts`
- `src/components/Marketplace.tsx`

قاعدة البيانات:
CREATE TABLE marketplace_policies (...)

```

---

## 📝 المرحلة 14: المهام المتبقية الصغيرة

### DIRECTIVE-067 إلى DIRECTIVE-112: مهام التحسين والصقل

```

المهام المتبقية (45 مهمة):

1. **Testing & Quality**:
   - كتابة unit tests شاملة (10 مهام)
   - integration tests (5 مهام)
   - E2E tests (3 مهام)
   - Performance benchmarks (2 مهام)

2. **Documentation**:
   - API documentation (5 مهام)
   - User guides (3 مهام)
   - Developer docs (2 مهام)
   - Video tutorials (2 مهام)

3. **UI/UX Improvements**:
   - Responsive design (2 مهام)
   - Dark mode (1 مهمة)
   - Accessibility (2 مهام)
   - Loading states (1 مهمة)

4. **Performance**:
   - Query optimization (2 مهام)
   - Caching strategies (2 مهام)
   - Bundle size reduction (1 مهمة)
   - CDN setup (1 مهمة)

5. **Security**:
   - Authentication/Authorization (2 مهام)
   - Rate limiting (1 مهمة)
   - Input validation (1 مهمة)
   - Audit logging (1 مهمة)

6. **DevOps**:
   - CI/CD pipelines (2 مهام)
   - Backup strategies (1 مهمة)
   - Disaster recovery (1 مهمة)
   - Cost monitoring (1 مهمة)

```

---

## 🎯 ملخص الأولويات

### الأولوية الفائقة (Critical Path - المرحلة 1):
- DIRECTIVE-018: Embeddings حقيقية
- DIRECTIVE-027: Reference Datasets
- DIRECTIVE-005: Parameterized Templates
- DIRECTIVE-006: Expand Mutation
- DIRECTIVE-019: Hill-Climbing Optimizer

### الأولوية العالية (المرحلة 2):
- DIRECTIVE-025: Test Suite Executor
- DIRECTIVE-026: Caching
- DIRECTIVE-052: Population Search
- DIRECTIVE-013: Hallucination Detection
- DIRECTIVE-030: Human Review UI

### الأولوية المتوسطة (المرحلة 3):
- DIRECTIVE-057: Reward Model
- DIRECTIVE-059: Pareto Optimization
- DIRECTIVE-060: A/B Testing
- DIRECTIVE-043: LangChain Pipelines

### الأولوية المنخفضة (المرحلة 4):
- DIRECTIVE-058: PPO Training
- DIRECTIVE-063: Continuous Learning
- DIRECTIVE-066: Marketplace

---

**ملاحظات نهائية**:
1. كل directive مستقل ويمكن تنفيذه بشكل منفصل
2. اتبع الأولويات لتحقيق أقصى قيمة
3. اختبر كل مكون قبل الانتقال للتالي
4. وثّق كل شيء أثناء التطوير
5. استخدم git commits واضحة
