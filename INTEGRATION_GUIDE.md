# Integration Guide - دليل التكامل الشامل

**كيفية دمج جميع مكونات Prompt Architect معاً**

---

## 📋 المحتويات

1. [نظرة عامة](#نظرة-عامة)
2. [الإعداد الأولي](#الإعداد-الأولي)
3. [سيناريوهات الاستخدام](#سيناريوهات-الاستخدام)
4. [أمثلة متكاملة](#أمثلة-متكاملة)
5. [أفضل الممارسات](#أفضل-الممارسات)
6. [استكشاف الأخطاء](#استكشاف-الأخطاء)

---

## 🎯 نظرة عامة

### سير العمل الكامل

```
مدخل: البرومبت الأصلي
    ↓
1. التصنيف (classifyPrompt)
    ↓
2. إنشاء Variations (mutations)
    ↓
3. التقييم الشامل (evaluators)
    ↓
4. الفلترة (balanceMetrics)
    ↓
5. الاختيار النهائي
    ↓
مخرج: أفضل برومبت
```

---

## ⚙️ الإعداد الأولي

### 1. تثبيت Dependencies

```bash
npm install
```

### 2. إعداد Environment Variables (اختياري)

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GROQ_API_KEY=gsk-...
```

### 3. Import الأساسية

```typescript
// Mutations
import {
  tryCatchStyleMutation,
  reduceContextMutation,
  expandMutation,
  constrainMutation
} from './src/mutations';

// Classification
import { classifyPrompt } from './src/types/promptTypes';

// Evaluation
import { evaluateSuggestions } from './src/evaluator';

// Balance Metrics
import {
  validateMetrics,
  BALANCED,
  QUALITY_FIRST,
  COST_OPTIMIZED
} from './src/config/balanceMetrics';

// Advanced Evaluators
import { calculateSemanticSimilarity } from './src/evaluator/semanticSimilarity';
import { detectHallucination } from './src/evaluator/hallucinationDetector';
import { FactualityChecker } from './src/evaluator/factualityChecker';
import { evaluateAgainstReference } from './src/evaluator/referenceMetrics';
```

---

## 📝 سيناريوهات الاستخدام

### السيناريو 1: استخدام أساسي (Quick Start)

**الهدف**: إنشاء وتقييم variations بسيطة

```typescript
async function basicUsage(originalPrompt: string) {
  // 1. إنشاء variations
  const variations = [
    tryCatchStyleMutation(originalPrompt),
    expandMutation(originalPrompt),
    reduceContextMutation(originalPrompt)
  ];

  // 2. تقييم
  const scored = await evaluateSuggestions(originalPrompt, variations);

  // 3. الحصول على الأفضل
  const best = scored[0];

  return {
    original: originalPrompt,
    improved: best.prompt,
    score: best.score,
    savings: calculateSavings(originalPrompt, best.prompt)
  };
}

// مثال
const result = await basicUsage("Write a function to validate email");
console.log('Improvement:', result.score);
console.log('Best:', result.improved);
```

---

### السيناريو 2: تحسين مُخصص حسب الفئة

**الهدف**: استخدام constraints مناسبة للفئة

```typescript
async function categoryOptimized(originalPrompt: string) {
  // 1. تصنيف البرومبت
  const classification = classifyPrompt(originalPrompt);
  console.log('Category:', classification.category);

  // 2. إنشاء variations مع constraints مناسبة
  const variations = [
    tryCatchStyleMutation(originalPrompt),
    expandMutation(originalPrompt),
    constrainMutation(originalPrompt, classification.category)
  ];

  // 3. تقييم
  const scored = await evaluateSuggestions(originalPrompt, variations);

  // 4. فلترة حسب فئة
  const categoryMetrics = getCategoryMetrics(classification.category);
  const filtered = scored.filter(s => {
    const metrics = extractMetrics(s);
    const validation = validateMetrics(metrics, categoryMetrics);
    return validation.isValid;
  });

  return filtered[0] || scored[0];
}

// مساعد: الحصول على معايير حسب الفئة
function getCategoryMetrics(category: string) {
  switch (category) {
    case 'CODE_GENERATION':
      return QUALITY_FIRST;
    case 'MARKETING_COPY':
      return COST_OPTIMIZED;
    default:
      return BALANCED;
  }
}
```

---

### السيناريو 3: تقييم متقدم مع Hallucination Detection

**الهدف**: كشف الهلوسة والتحقق من الجودة

```typescript
async function advancedEvaluation(
  originalPrompt: string,
  provider: any
) {
  // 1. إنشاء variations
  const variations = [
    tryCatchStyleMutation(originalPrompt),
    expandMutation(originalPrompt)
  ];

  // 2. تقييم أساسي
  const scored = await evaluateSuggestions(originalPrompt, variations);

  // 3. فحص الهلوسة لكل variation
  const withHallucinationCheck = await Promise.all(
    scored.map(async (suggestion) => {
      // محاكاة مخرج
      const mockOutput = await simulateOutput(suggestion.prompt, provider);

      // كشف الهلوسة
      const hallucinationScore = await detectHallucination(
        suggestion.prompt,
        mockOutput,
        provider
      );

      return {
        ...suggestion,
        hallucinationRisk: hallucinationScore.score,
        hallucinationSeverity: getHallucinationSeverity(hallucinationScore.score),
        isReliable: hallucinationScore.score < 0.3
      };
    })
  );

  // 4. فلترة الموثوقة فقط
  const reliable = withHallucinationCheck.filter(s => s.isReliable);

  // 5. ترتيب حسب النقاط
  reliable.sort((a, b) => b.score - a.score);

  return reliable[0];
}

// مساعد: محاكاة مخرج
async function simulateOutput(prompt: string, provider: any): Promise<string> {
  // في الإنتاج: استدعاء LLM حقيقي
  return `Mock output for: ${prompt}`;
}
```

---

### السيناريو 4: RAG-based Factuality Checking

**الهدف**: التحقق من صحة المعلومات

```typescript
async function factualityOptimized(
  originalPrompt: string,
  knowledgeBase: Document[]
) {
  // 1. إعداد Factuality Checker
  const checker = new FactualityChecker({
    vectorStore: {
      provider: 'memory',
      dimension: 384,
      metric: 'cosine'
    },
    embeddingProvider: {
      type: 'mock',
      dimension: 384
    },
    requireMultipleSources: true,
    minSourceCount: 2
  });

  // 2. إضافة قاعدة المعرفة
  const vectorStore = checker.getVectorStore();
  for (const doc of knowledgeBase) {
    await vectorStore.addDocument(doc);
  }

  // 3. إنشاء variations
  const variations = [
    tryCatchStyleMutation(originalPrompt),
    expandMutation(originalPrompt)
  ];

  // 4. تقييم أساسي
  const scored = await evaluateSuggestions(originalPrompt, variations);

  // 5. فحص الصحة لكل variation
  const withFactCheck = await Promise.all(
    scored.map(async (suggestion) => {
      const mockOutput = await simulateOutput(suggestion.prompt, null);
      const factCheck = await checker.verifyFactuality(mockOutput);

      return {
        ...suggestion,
        factualityScore: factCheck.overallScore,
        isFactual: factCheck.isFactual,
        sources: factCheck.sources,
        confidence: factCheck.confidence
      };
    })
  );

  // 6. فلترة الصحيحة فقط
  const factual = withFactCheck.filter(s => s.isFactual);

  // 7. ترتيب حسب الجودة
  factual.sort((a, b) => b.factualityScore - a.factualityScore);

  return factual[0];
}
```

---

### السيناريو 5: Complete Pipeline (الأقوى)

**الهدف**: دمج جميع الميزات معاً

```typescript
async function completePipeline(
  originalPrompt: string,
  options: {
    provider: any;
    knowledgeBase?: Document[];
    referenceOutputs?: string[];
    preset: 'cost' | 'quality' | 'balanced';
  }
) {
  console.log('🚀 Starting Complete Pipeline...\n');

  // === المرحلة 1: التصنيف ===
  console.log('📊 Step 1: Classification');
  const classification = classifyPrompt(originalPrompt);
  console.log(`  Category: ${classification.category}`);
  console.log(`  Confidence: ${(classification.confidence * 100).toFixed(1)}%\n`);

  // === المرحلة 2: إنشاء Variations ===
  console.log('🔄 Step 2: Generating Variations');
  const variations = [
    tryCatchStyleMutation(originalPrompt),
    reduceContextMutation(originalPrompt),
    expandMutation(originalPrompt),
    constrainMutation(originalPrompt, classification.category)
  ];
  console.log(`  Generated: ${variations.length} variations\n`);

  // === المرحلة 3: التقييم الأساسي ===
  console.log('📈 Step 3: Basic Evaluation');
  const scored = await evaluateSuggestions(originalPrompt, variations);
  console.log(`  Scored: ${scored.length} variations\n`);

  // === المرحلة 4: Hallucination Detection ===
  console.log('🔍 Step 4: Hallucination Detection');
  const withHallucination = await Promise.all(
    scored.map(async (suggestion) => {
      const mockOutput = await simulateOutput(suggestion.prompt, options.provider);
      const hallucinationScore = await detectHallucination(
        suggestion.prompt,
        mockOutput,
        options.provider
      );

      return {
        ...suggestion,
        hallucination: hallucinationScore
      };
    })
  );
  console.log(`  Checked: ${withHallucination.length} outputs\n`);

  // === المرحلة 5: Factuality Check (if knowledge base provided) ===
  let withFactuality = withHallucination;
  if (options.knowledgeBase) {
    console.log('✓ Step 5: Factuality Verification');
    const checker = new FactualityChecker({
      vectorStore: { provider: 'memory', dimension: 384, metric: 'cosine' },
      embeddingProvider: { type: 'mock', dimension: 384 }
    });

    const vectorStore = checker.getVectorStore();
    for (const doc of options.knowledgeBase) {
      await vectorStore.addDocument(doc);
    }

    withFactuality = await Promise.all(
      withHallucination.map(async (s) => {
        const mockOutput = await simulateOutput(s.prompt, options.provider);
        const factCheck = await checker.verifyFactuality(mockOutput);
        return { ...s, factuality: factCheck };
      })
    );
    console.log(`  Verified: ${withFactuality.length} outputs\n`);
  }

  // === المرحلة 6: Reference Comparison (if references provided) ===
  let withReference = withFactuality;
  if (options.referenceOutputs) {
    console.log('📊 Step 6: Reference Comparison');
    withReference = await Promise.all(
      withFactuality.map(async (s) => {
        const mockOutput = await simulateOutput(s.prompt, options.provider);
        const refMetrics = evaluateAgainstReference(
          s.prompt,
          mockOutput,
          options.referenceOutputs!
        );
        return { ...s, reference: refMetrics };
      })
    );
    console.log(`  Compared: ${withReference.length} outputs\n`);
  }

  // === المرحلة 7: Balance Metrics Validation ===
  console.log('⚖️  Step 7: Balance Metrics');
  const preset = {
    'cost': COST_OPTIMIZED,
    'quality': QUALITY_FIRST,
    'balanced': BALANCED
  }[options.preset];

  const validated = withReference.map((s) => {
    const metrics = {
      quality: s.score / 100,
      cost: s.estimatedCost,
      latency: s.latency || 2000,
      hallucinationRate: s.hallucination?.score || 0,
      similarity: s.similarity
    };

    const validation = validateMetrics(metrics, preset);

    return {
      ...s,
      validation,
      finalScore: validation.score
    };
  });

  // === المرحلة 8: الفلترة والترتيب النهائي ===
  console.log('🎯 Step 8: Final Filtering & Ranking');
  const valid = validated.filter(s => s.validation.isValid);
  valid.sort((a, b) => b.finalScore - a.finalScore);

  const best = valid[0] || validated[0];

  // === النتيجة النهائية ===
  console.log('\n✅ Pipeline Complete!\n');
  console.log('=' 60);
  console.log('Best Variation:');
  console.log('  Score:', best.finalScore);
  console.log('  Cost:', `$${best.estimatedCost.toFixed(4)}`);
  console.log('  Hallucination Risk:', `${(best.hallucination?.score * 100 || 0).toFixed(1)}%`);
  if (best.factuality) {
    console.log('  Factuality:', `${best.factuality.overallScore.toFixed(1)}/100`);
  }
  if (best.reference) {
    console.log('  ROUGE-L:', `${(best.reference.rouge.rougeL.f1 * 100).toFixed(1)}%`);
  }
  console.log('=' * 60);

  return best;
}

// استخدام
const result = await completePipeline(
  "Explain how React works",
  {
    provider: { name: 'openai', supportsLogprobs: true },
    knowledgeBase: trustedReactDocs,
    referenceOutputs: highQualityExplanations,
    preset: 'quality'
  }
);
```

---

## 🎯 أمثلة متكاملة

### مثال 1: تحسين Code Generation Prompt

```typescript
async function optimizeCodePrompt(prompt: string) {
  // التصنيف
  const classification = classifyPrompt(prompt);
  console.assert(classification.category === 'CODE_GENERATION');

  // إنشاء variations
  const variations = [
    // Try/Catch للتعامل مع الأخطاء
    tryCatchStyleMutation(prompt),

    // Expand لإضافة تفاصيل
    expandMutation(prompt),

    // Constraints خاصة بالكود
    constrainMutation(prompt, 'CODE_GENERATION')
  ];

  // تقييم مع التركيز على الجودة
  const scored = await evaluateSuggestions(prompt, variations);

  // فلترة بمعايير QUALITY_FIRST
  const filtered = scored.filter(s => {
    const metrics = {
      quality: s.score / 100,
      cost: s.estimatedCost,
      latency: s.latency || 2000,
      hallucinationRate: 0.05, // منخفض للكود
      similarity: s.similarity
    };
    return validateMetrics(metrics, QUALITY_FIRST).isValid;
  });

  return filtered[0];
}

// استخدام
const improved = await optimizeCodePrompt(
  "Write a function to validate email"
);
console.log(improved.prompt);
// Expected output: Detailed, clear, with examples and success criteria
```

---

### مثال 2: تحسين Marketing Copy

```typescript
async function optimizeMarketingCopy(prompt: string) {
  // التصنيف
  const classification = classifyPrompt(prompt);
  console.assert(classification.category === 'MARKETING_COPY');

  // Variations مع تركيز على الإيجاز
  const variations = [
    // Reduce للتقليل من التكلفة
    reduceContextMutation(prompt),

    // Constraints تسويقية
    constrainMutation(prompt, 'MARKETING_COPY')
  ];

  // تقييم مع COST_OPTIMIZED
  const scored = await evaluateSuggestions(prompt, variations);

  // فلترة بمعايير التكلفة
  const filtered = scored.filter(s => {
    const metrics = {
      quality: s.score / 100,
      cost: s.estimatedCost,
      latency: s.latency || 1500, // سريع
      hallucinationRate: 0.15, // مقبول
      similarity: s.similarity
    };
    return validateMetrics(metrics, COST_OPTIMIZED).isValid;
  });

  return filtered[0];
}
```

---

## 🔧 أفضل الممارسات

### 1. اختيار الـ Mutations المناسبة

```typescript
function selectMutations(category: string): Array<(prompt: string) => PromptVariation> {
  const baseMutations = [tryCatchStyleMutation];

  switch (category) {
    case 'CODE_GENERATION':
      return [...baseMutations, expandMutation]; // تفاصيل أكثر

    case 'MARKETING_COPY':
      return [...baseMutations, reduceContextMutation]; // إيجاز

    case 'CONTENT_WRITING':
      return [...baseMutations, expandMutation]; // محتوى غني

    default:
      return baseMutations;
  }
}
```

### 2. استخدام الـ Caching بكفاءة

```typescript
import { getCacheStats, clearEmbeddingCache } from './src/evaluator/semanticSimilarity';

// فحص الـ cache
const stats = getCacheStats();
console.log('Cache size:', stats.size);
console.log('Providers:', stats.providers);

// تنظيف الـ cache عند الحاجة
if (stats.size > 1000) {
  clearEmbeddingCache();
  console.log('Cache cleared');
}
```

### 3. معالجة الأخطاء

```typescript
async function safeEvaluation(prompt: string) {
  try {
    const variations = [
      tryCatchStyleMutation(prompt),
      expandMutation(prompt)
    ];

    const scored = await evaluateSuggestions(prompt, variations);
    return scored[0];

  } catch (error) {
    console.error('Evaluation failed:', error);

    // Fallback: استخدام البرومبت الأصلي
    return {
      prompt,
      score: 50,
      mutation: 'none',
      error: error.message
    };
  }
}
```

---

## 🐛 استكشاف الأخطاء

### المشكلة 1: Embeddings بطيئة

**الحل**:
```typescript
// استخدم mock provider للتطوير
const provider = createMockProvider(384);

// أو استخدم caching
const similarity = await calculateSemanticSimilarity(
  text1,
  text2,
  provider,
  true // useCache = true
);
```

### المشكلة 2: نقاط منخفضة

**الحل**:
```typescript
// فحص السبب
const validation = validateMetrics(metrics, preset);
console.log('Violations:', validation.violations);
console.log('Passed:', validation.passed);

// تعديل الـ preset
const customPreset = createCustomMetrics('balanced', {
  minQuality: 0.6, // خفّض المعايير
  maxCost: 0.05
});
```

### المشكلة 3: Hallucination score عالي

**الحل**:
```typescript
// استخدم expand mutation لمزيد من الوضوح
const expanded = expandMutation(prompt);

// أو أضف context
const withContext = `${prompt}\n\nContext: ${relevantInfo}`;

// أعد الفحص
const score = await detectHallucination(
  withContext,
  output,
  provider,
  relevantInfo
);
```

---

## 📚 موارد إضافية

- [API Reference](README.md#api-reference)
- [Balance Metrics Guide](src/config/README.md)
- [Mutation Examples](src/mutations.examples.md)
- [Implementation Status](IMPLEMENTATION_STATUS_DETAILED.md)

---

## ✅ Checklist للتكامل

- [ ] تثبيت Dependencies
- [ ] إعداد Environment Variables
- [ ] اختبار Mutations الأساسية
- [ ] تجربة Classification
- [ ] اختبار Evaluation Pipeline
- [ ] فحص Balance Metrics
- [ ] تجربة Hallucination Detection
- [ ] تجربة Factuality Checking
- [ ] قياس الأداء
- [ ] معالجة الأخطاء

---

**آخر تحديث**: 2025-12-14
**النسخة**: 1.0.0

**جاهز للإنتاج!** 🚀
