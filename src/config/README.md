# Balance Metrics Configuration

نظام شامل لتحديد ومراقبة معايير التوازن بين الجودة والتكلفة والسرعة والموثوقية في تحسين البرومبتات.

## 📋 المحتويات

- [نظرة عامة](#نظرة-عامة)
- [المعايير الأساسية](#المعايير-الأساسية)
- [الإعدادات المسبقة](#الإعدادات-المسبقة)
- [الاستخدام](#الاستخدام)
- [أمثلة](#أمثلة)

---

## نظرة عامة

يوفر هذا النظام طريقة منظمة لتحديد الحدود المقبولة للمقاييس المختلفة عند تقييم variations البرومبتات:

- **الجودة/الدقة** (Quality/Accuracy): ما مدى جودة النتائج؟
- **التكلفة** (Cost): ما تكلفة كل طلب؟
- **الزمن** (Latency): ما سرعة الاستجابة؟
- **الموثوقية** (Reliability): ما معدل الهلوسة/الأخطاء؟

## المعايير الأساسية

### BalanceMetrics Interface

```typescript
interface BalanceMetrics {
  minQuality: number;           // 0-1, الحد الأدنى للجودة
  maxCost: number;              // بالدولار، الحد الأقصى للتكلفة
  maxLatency: number;           // بالميليثانية، الحد الأقصى للزمن
  maxHallucinationRate: number; // 0-1, الحد الأقصى للهلوسة
  minSimilarity: number;        // 0-1, الحد الأدنى للتشابه
  weights: MetricWeights;       // أوزان كل معيار
}
```

### MetricWeights Interface

```typescript
interface MetricWeights {
  quality: number;      // وزن الجودة
  cost: number;         // وزن التكلفة
  latency: number;      // وزن السرعة
  reliability: number;  // وزن الموثوقية
}
```

**ملاحظة**: يجب أن يكون مجموع الأوزان = 1.0

---

## الإعدادات المسبقة

### 1. COST_OPTIMIZED
**الاستخدام**: التطبيقات ذات الحجم الكبير حيث التكلفة حرجة

```typescript
{
  minQuality: 0.6,
  maxCost: 0.01,      // $0.01 فقط
  maxLatency: 5000,   // 5 ثوان
  maxHallucinationRate: 0.2,
  minSimilarity: 0.5,
  weights: {
    quality: 0.2,
    cost: 0.5,        // 50% تركيز على التكلفة
    latency: 0.15,
    reliability: 0.15,
  }
}
```

### 2. QUALITY_FIRST
**الاستخدام**: العمليات الحرجة حيث الجودة أهم من التكلفة

```typescript
{
  minQuality: 0.9,    // جودة عالية جداً
  maxCost: 0.1,       // $0.10 مقبول
  maxLatency: 10000,  // 10 ثوان
  maxHallucinationRate: 0.05,  // 5% فقط
  minSimilarity: 0.8,
  weights: {
    quality: 0.5,     // 50% تركيز على الجودة
    cost: 0.1,
    latency: 0.15,
    reliability: 0.25,
  }
}
```

### 3. BALANCED
**الاستخدام**: الاستخدام العام، توازن بين جميع المعايير

```typescript
{
  minQuality: 0.75,
  maxCost: 0.03,
  maxLatency: 3000,   // 3 ثوان
  maxHallucinationRate: 0.1,
  minSimilarity: 0.7,
  weights: {
    quality: 0.3,
    cost: 0.3,
    latency: 0.2,
    reliability: 0.2,
  }
}
```

### 4. SPEED_OPTIMIZED
**الاستخدام**: التطبيقات الفورية real-time

```typescript
{
  minQuality: 0.65,
  maxCost: 0.02,
  maxLatency: 1500,   // 1.5 ثانية فقط
  maxHallucinationRate: 0.15,
  minSimilarity: 0.6,
  weights: {
    quality: 0.2,
    cost: 0.2,
    latency: 0.45,    // 45% تركيز على السرعة
    reliability: 0.15,
  }
}
```

---

## الاستخدام

### الاستخدام الأساسي

```typescript
import { validateMetrics, BALANCED } from './config/balanceMetrics';

// مقاييس الاقتراح
const suggestionMetrics = {
  quality: 0.85,
  cost: 0.025,
  latency: 2500,
  hallucinationRate: 0.08,
  similarity: 0.75,
};

// التحقق من صحة الاقتراح
const result = validateMetrics(suggestionMetrics, BALANCED);

console.log(result.isValid);        // true/false
console.log(result.score);          // 0-100
console.log(result.violations);     // قائمة المخالفات
console.log(result.recommendation); // توصية
```

### استخدام preset محدد

```typescript
import { getPreset } from './config/balanceMetrics';

const costOptimized = getPreset('cost-optimized');
const result = validateMetrics(suggestionMetrics, costOptimized);
```

### إنشاء معايير مخصصة

```typescript
import { createCustomMetrics } from './config/balanceMetrics';

// ابدأ من BALANCED وخصص
const customMetrics = createCustomMetrics('balanced', {
  minQuality: 0.8,      // جودة أعلى
  maxCost: 0.02,        // تكلفة أقل
  weights: {
    quality: 0.4,       // زد وزن الجودة
    cost: 0.35,         // زد وزن التكلفة
    latency: 0.15,
    reliability: 0.1,
  },
});
```

### حساب النقاط فقط

```typescript
import { calculateWeightedScore } from './config/balanceMetrics';

const score = calculateWeightedScore(suggestionMetrics, BALANCED);
console.log(`Score: ${score}/100`);
```

---

## أمثلة

### مثال 1: فلترة الاقتراحات

```typescript
import { validateMetrics, QUALITY_FIRST } from './config/balanceMetrics';

function filterValidSuggestions(suggestions: ScoredSuggestion[]) {
  return suggestions.filter(suggestion => {
    const metrics = {
      quality: suggestion.score / 100,
      cost: suggestion.estimatedCost,
      latency: 2000, // أو القيمة الفعلية
      hallucinationRate: 0.05, // من hallucination detector
      similarity: suggestion.similarity,
    };

    const validation = validateMetrics(metrics, QUALITY_FIRST);
    return validation.isValid;
  });
}
```

### مثال 2: ترتيب الاقتراحات حسب النقاط

```typescript
import { calculateWeightedScore, BALANCED } from './config/balanceMetrics';

function rankSuggestions(suggestions: ScoredSuggestion[]) {
  return suggestions
    .map(suggestion => ({
      ...suggestion,
      balanceScore: calculateWeightedScore(
        {
          quality: suggestion.score / 100,
          cost: suggestion.estimatedCost,
          latency: 2000,
          hallucinationRate: 0.05,
          similarity: suggestion.similarity,
        },
        BALANCED
      ),
    }))
    .sort((a, b) => b.balanceScore - a.balanceScore);
}
```

### مثال 3: عرض التحذيرات للمستخدم

```typescript
import { validateMetrics, BALANCED } from './config/balanceMetrics';

function getSuggestionWarnings(suggestion: ScoredSuggestion) {
  const metrics = {
    quality: suggestion.score / 100,
    cost: suggestion.estimatedCost,
    latency: 2000,
    hallucinationRate: 0.05,
    similarity: suggestion.similarity,
  };

  const validation = validateMetrics(metrics, BALANCED);

  if (!validation.isValid) {
    return {
      hasWarnings: true,
      violations: validation.violations.map(v => ({
        severity: v.severity,
        message: v.message,
      })),
      recommendation: validation.recommendation,
    };
  }

  return { hasWarnings: false };
}
```

### مثال 4: معايير حسب فئة البرومبت

```typescript
import { createCustomMetrics } from './config/balanceMetrics';

function getMetricsForCategory(category: string) {
  switch (category) {
    case 'CODE_GENERATION':
      // الكود يحتاج جودة عالية وموثوقية
      return createCustomMetrics('quality-first', {
        weights: {
          quality: 0.45,
          cost: 0.15,
          latency: 0.15,
          reliability: 0.25,
        },
      });

    case 'MARKETING_COPY':
      // التسويق يحتاج سرعة وتكلفة منخفضة
      return createCustomMetrics('speed-optimized', {
        weights: {
          quality: 0.25,
          cost: 0.3,
          latency: 0.3,
          reliability: 0.15,
        },
      });

    case 'CONTENT_WRITING':
      // المحتوى متوازن
      return createCustomMetrics('balanced', {});

    default:
      return createCustomMetrics('balanced', {});
  }
}
```

---

## ValidationResult Structure

عند استدعاء `validateMetrics()`، تحصل على:

```typescript
{
  isValid: boolean;              // هل يلبي جميع المعايير؟
  score: number;                 // النقاط الإجمالية (0-100)
  violations: [                  // قائمة المخالفات
    {
      metric: 'maxCost',
      threshold: 0.03,
      actual: 0.045,
      severity: 'medium',
      message: 'Cost $0.0450 exceeds maximum $0.0300'
    }
  ],
  passed: ['quality', 'latency', 'reliability'],  // المعايير التي نجحت
  recommendation: 'Moderate issues with: maxCost. Review carefully before accepting.'
}
```

---

## Severity Levels

المخالفات لها 3 مستويات خطورة:

- **low**: الانحراف < 10%
- **medium**: الانحراف 10-30%
- **high**: الانحراف > 30%

---

## Best Practices

### 1. اختر الـ preset المناسب لحالتك

```typescript
// إنتاج عالي الحجم
const metrics = COST_OPTIMIZED;

// تطبيقات حرجة
const metrics = QUALITY_FIRST;

// تطبيقات real-time
const metrics = SPEED_OPTIMIZED;
```

### 2. خصص الأوزان حسب الأولويات

```typescript
const customMetrics = createCustomMetrics('balanced', {
  weights: {
    quality: 0.5,    // أولوية عالية للجودة
    cost: 0.2,
    latency: 0.2,
    reliability: 0.1,
  },
});
```

### 3. راقب النتائج وعدّل

```typescript
// تتبع معدل القبول
const acceptRate = acceptedSuggestions / totalSuggestions;

// إذا كان معدل القبول منخفض، خفف المعايير
if (acceptRate < 0.3) {
  customMetrics.minQuality -= 0.05;
  customMetrics.maxCost += 0.005;
}
```

### 4. استخدم معايير مختلفة لمراحل مختلفة

```typescript
// في exploration: استخدم معايير أكثر تساهلاً
const explorationMetrics = COST_OPTIMIZED;

// في production: استخدم معايير صارمة
const productionMetrics = QUALITY_FIRST;
```

---

## Integration with Evaluator

دمج مع نظام التقييم الحالي:

```typescript
// في evaluator.ts
import { validateMetrics, BALANCED } from './config/balanceMetrics';

export function evaluateSuggestions(
  originalPrompt: string,
  variations: PromptVariation[]
): ScoredSuggestion[] {
  const scored = variations.map(variation => {
    // ... حسابات موجودة ...

    // إضافة التحقق من balance metrics
    const balanceValidation = validateMetrics(
      {
        quality: score / 100,
        cost: estimatedCost,
        latency: 2000,
        hallucinationRate: 0.05,
        similarity,
      },
      BALANCED
    );

    return {
      prompt: variation.prompt,
      mutation: variation.mutation,
      score,
      tokenCount,
      estimatedCost,
      similarity,
      // إضافة معلومات التحقق
      balanceScore: balanceValidation.score,
      meetsBalanceCriteria: balanceValidation.isValid,
      balanceViolations: balanceValidation.violations,
    };
  });

  return scored.sort((a, b) => b.balanceScore - a.balanceScore);
}
```

---

## TypeScript Types

جميع الـ types متوفرة للـ autocomplete:

```typescript
import type {
  BalanceMetrics,
  MetricWeights,
  ValidationResult,
  MetricViolation,
  SuggestionMetrics,
  PresetType,
} from './config/balanceMetrics';
```

---

## Testing

شغّل الأمثلة:

```bash
ts-node src/config/balanceMetrics.example.ts
```

---

## Future Enhancements

- [ ] إضافة معايير لـ context window usage
- [ ] إضافة معايير لـ output length
- [ ] تكامل مع A/B testing results
- [ ] تعلم آلي لتحسين الأوزان تلقائياً
- [ ] Dashboard لعرض توزيع المقاييس

---

**تم التطوير في**: المرحلة 0 - الأساسيات
**DIRECTIVE**: DIRECTIVE-001
**الملف**: `src/config/balanceMetrics.ts`
