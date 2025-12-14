# 📊 Overfitting Detection System (DIRECTIVE-038)

> **نظام متكامل للكشف عن ومعالجة Overfitting في البرومبتات المُحسّنة**

## 🎯 الهدف

التأكد من أن البرومبتات المُحسّنة تعمل بشكل جيد على مدخلات متنوعة، وليس فقط على بيانات التدريب التي تم تحسينها عليها.

## 🔍 المشكلة

عندما نقوم بتحسين البرومبتات، قد نحصل على نتائج ممتازة على بيانات التدريب، لكن الأداء ينخفض بشكل كبير على بيانات جديدة. هذه ظاهرة **Overfitting**.

### مثال على Overfitting:

```typescript
// برومبت overfitted
const overfittedPrompt = `
You are an expert code generator.
When writing a sorting function, use these exact steps:
1. Check if array is [3, 1, 4] → return [1, 3, 4]
2. Check if array is [5, 2, 8] → return [2, 5, 8]
...50 more specific examples...
`;

// نتيجة:
// ✅ Training: 98% accuracy (على الأمثلة المحددة)
// ❌ Validation: 45% accuracy (على بيانات جديدة)
```

## 🛠️ الحل: 4 استراتيجيات متكاملة

### 1️⃣ Diverse Test Sets
اختبار على بيانات متنوعة من فئات مختلفة.

```typescript
import { analyzeDiversity } from './overfittingDetector';

const testCases = loadTestCases();
const diversity = analyzeDiversity(testCases);

console.log(`Diversity Score: ${diversity.diversityScore}`);
console.log(`Categories: ${diversity.uniqueCategories}`);
console.log(`Sufficient: ${diversity.isSufficientlyDiverse}`);
```

### 2️⃣ K-Fold Cross Validation
تقسيم البيانات إلى K أجزاء والتحقق من استقرار الأداء.

```typescript
import { kFoldCrossValidation } from './overfittingDetector';

const result = await kFoldCrossValidation(
    prompt,
    testCases,
    llmExecutor,
    5 // 5-fold
);

console.log(`Mean Score: ${result.meanScore}`);
console.log(`Std Deviation: ${result.stdDeviation}`);
console.log(`Stable: ${result.isStable}`);
```

### 3️⃣ Held-out Validation
الاحتفاظ بمجموعة اختبار منفصلة لم يتم استخدامها في التدريب.

```typescript
import { heldOutValidation } from './overfittingDetector';

const result = await heldOutValidation(prompt, testCases, executor);

console.log(`Train Score: ${result.trainScore}`);
console.log(`Val Score: ${result.valScore}`);
console.log(`Test Score: ${result.testScore}`);
console.log(`Gap: ${result.trainTestGap}`);
```

### 4️⃣ Regularization
معاقبة التعقيد الزائد وتبسيط البرومبتات المعقدة.

```typescript
import { calculateRegularization, simplifyPrompt } from './overfittingDetector';

const penalty = calculateRegularization(complexPrompt);
console.log(`Regularization Penalty: ${penalty}`);

if (penalty > 0.5) {
    const simplified = simplifyPrompt(complexPrompt, 0.3); // تقليل 30%
    console.log('Simplified Prompt:', simplified);
}
```

## 🚀 الاستخدام السريع

### الكشف الأساسي

```typescript
import { detectOverfitting } from './overfittingDetector';
import { executeTestSuite } from '../sandbox/testExecutor';

// 1. تقسيم البيانات
const trainCases = allTestCases.slice(0, 60);
const valCases = allTestCases.slice(60);

// 2. تشغيل الاختبارات
const trainResults = await executeTestSuite([prompt], trainCases, executor);
const valResults = await executeTestSuite([prompt], valCases, executor);

// 3. كشف Overfitting
const report = await detectOverfitting(
    prompt,
    trainResults[0],
    valResults[0]
);

// 4. عرض النتائج
console.log(`Overfitted: ${report.isOverfit}`);
console.log(`Severity: ${report.severity}`);
console.log(`Gap: ${(report.gap * 100).toFixed(1)}%`);
console.log(`Recommendation: ${report.recommendation}`);
```

### التحليل الشامل

```typescript
import { comprehensiveOverfittingAnalysis } from './overfittingDetector';

const analysis = await comprehensiveOverfittingAnalysis(
    prompt,
    testCases,
    executor
);

// يتضمن:
// - Overfitting Report
// - K-Fold Cross Validation
// - Held-out Validation
// - Diversity Analysis
// - Regularization Penalty
// - Simplified Prompt (إذا لزم الأمر)

console.log('Overfitting:', analysis.overfittingReport.isOverfit);
console.log('Cross-Val Stable:', analysis.crossValidation.isStable);
console.log('Generalization:', analysis.heldOutValidation.generalizationScore);
console.log('Diversity:', analysis.diversityAnalysis.diversityScore);
```

### عرض التقرير

```typescript
import { printOverfittingReport } from './overfittingDetector';

printOverfittingReport(report);
```

**المخرجات:**

```
============================================================
📊 تقرير كشف Overfitting
============================================================

🎯 النتيجة: ⚠️ OVERFITTED
📈 الشدة: MODERATE
🎲 الثقة: 87.3%

📊 النقاط:
  • Training Score:   94.2%
  • Validation Score: 71.5%
  • Gap:              22.7%

🔍 تحليل التباين:
  • Train Variance:   0.0023
  • Val Variance:     0.0156
  • Variance Ratio:   6.78

🧩 تحليل التعقيد:
  • Prompt Length:    3542 chars
  • Token Count:      886 tokens
  • Complexity Score: 88.6%
  • Too Complex:      YES ⚠️

💡 التوصية:
  ⚠️ Overfitting متوسط: تعديلات مهمة مطلوبة
  • قلل تخصيص البرومبت لبيانات التدريب (الفجوة: 22.7%)
  • أضف المزيد من البيانات المتنوعة للتدريب
  • بسّط البرومبت (حالياً 886 tokens)
  • أزل الأمثلة أو الشروح الزائدة
  ...
============================================================
```

## 📚 الواجهات الرئيسية (Types)

### OverfittingReport

```typescript
interface OverfittingReport {
    isOverfit: boolean;
    trainScore: number;
    valScore: number;
    gap: number;
    confidence: number;
    severity: 'none' | 'mild' | 'moderate' | 'severe';
    recommendation: string;
    analysis: OverfittingAnalysis;
}
```

### OverfittingConfig

```typescript
interface OverfittingConfig {
    gapThreshold?: number;           // default: 0.1
    minAcceptableScore?: number;     // default: 0.7
    maxComplexityTokens?: number;    // default: 1000
    maxVarianceRatio?: number;       // default: 2.0
    enableDetailedAnalysis?: boolean; // default: true
}
```

### CrossValidationResult

```typescript
interface CrossValidationResult {
    folds: number;
    foldScores: number[];
    meanScore: number;
    stdDeviation: number;
    bestFold: number;
    worstFold: number;
    isStable: boolean;
}
```

### DiversityAnalysis

```typescript
interface DiversityAnalysis {
    diversityScore: number;
    uniqueCategories: number;
    categoryDistribution: Map<string, number>;
    isSufficientlyDiverse: boolean;
}
```

## 🎨 أمثلة عملية

### مثال 1: كشف Overfitting في برومبت توليد الكود

```typescript
const codePrompt = `
You are an expert Python developer.
Generate sorting algorithms with these specifications:
- Time complexity: O(n log n)
- Space complexity: O(1) or O(log n)
- Handle edge cases: empty array, single element, duplicates
- Include comprehensive docstrings
- Add type hints
- Write unit tests
`;

const testCases = [
    { id: '1', prompt: 'Sort [3,1,4,1,5]', ... },
    { id: '2', prompt: 'Sort []', ... },
    { id: '3', prompt: 'Sort [1]', ... },
    // ... more diverse cases
];

const report = await detectOverfitting(
    codePrompt,
    trainResults,
    valResults
);

if (report.isOverfit) {
    console.log('⚠️ Overfitting detected!');
    console.log(report.recommendation);
    
    // Apply simplification
    const simplified = simplifyPrompt(codePrompt, 0.3);
    console.log('Simplified:', simplified);
}
```

### مثال 2: Cross Validation للتحقق من الاستقرار

```typescript
const prompt = "Generate marketing copy for tech products";
const testCases = loadMarketingTestCases(); // 50 cases

const cvResult = await kFoldCrossValidation(prompt, testCases, executor, 10);

if (!cvResult.isStable) {
    console.log('⚠️ Results are not stable across folds');
    console.log(`Std Dev: ${cvResult.stdDeviation}`);
    console.log('Consider:');
    console.log('- Adding more constraints to the prompt');
    console.log('- Using more consistent examples');
    console.log('- Increasing prompt specificity');
}
```

### مثال 3: تحليل تنوع البيانات

```typescript
const testCases = loadAllTestCases();
const diversity = analyzeDiversity(testCases);

console.log(`Categories: ${diversity.uniqueCategories}`);
console.log(`Diversity: ${diversity.diversityScore.toFixed(2)}`);

if (!diversity.isSufficientlyDiverse) {
    console.warn('⚠️ Dataset is not diverse enough!');
    console.log('Current distribution:');
    
    diversity.categoryDistribution.forEach((count, category) => {
        const percentage = (count / testCases.length * 100).toFixed(1);
        console.log(`  ${category}: ${count} (${percentage}%)`);
    });
    
    console.log('\nRecommendation: Add more test cases from underrepresented categories');
}
```

## 🔬 التحليل العلمي

### كيف يتم حساب Overfitting؟

```typescript
// 1. Gap Analysis
const gap = trainScore - valScore;
const isOverfit_gap = gap > threshold; // default: 0.1

// 2. Variance Ratio
const varianceRatio = valVariance / trainVariance;
const isOverfit_variance = varianceRatio > maxRatio; // default: 2.0

// 3. Complexity Check
const tokenCount = estimateTokenCount(prompt);
const isOverfit_complexity = tokenCount > maxTokens; // default: 1000

// 4. Final Decision
const isOverfit = isOverfit_gap || isOverfit_variance || isOverfit_complexity;
```

### Severity Levels

| Severity | Gap | Variance Ratio | Action |
|----------|-----|----------------|--------|
| **None** | < 10% | < 2.0 | ✅ No action needed |
| **Mild** | 10-20% | 2.0-3.0 | ⚡ Minor adjustments |
| **Moderate** | 20-30% | 3.0-4.0 | ⚠️ Significant changes needed |
| **Severe** | > 30% | > 4.0 | 🚨 Complete redesign required |

### Regularization Penalty

```typescript
// L1 Penalty: معاقبة الطول
const l1 = promptLength * lambda;

// L2 Penalty: معاقبة التعقيد
const l2 = Math.pow(tokenCount, 2) * lambda;

// Total Penalty
const penalty = l1 + l2;
```

## 🧪 الاختبار

```bash
# تشغيل الاختبارات
npm test -- overfittingDetector.test.ts

# تشغيل العروض التوضيحية
npm run demo:overfitting
```

## 📈 أفضل الممارسات

### ✅ DO's

1. **اختبر على بيانات متنوعة**
   ```typescript
   const testCases = [
       ...codeGenerationCases,
       ...contentWritingCases,
       ...dataAnalysisCases
   ];
   ```

2. **استخدم K-Fold للتحقق من الاستقرار**
   ```typescript
   const cv = await kFoldCrossValidation(prompt, testCases, executor, 5);
   ```

3. **احتفظ بـ test set منفصل**
   ```typescript
   const { train, validation, test } = splitDataset(allCases);
   // Never use test set during optimization!
   ```

4. **راقب التعقيد**
   ```typescript
   const penalty = calculateRegularization(prompt);
   if (penalty > 0.5) simplifyPrompt(prompt);
   ```

### ❌ DON'Ts

1. **لا تختبر فقط على نوع واحد من البيانات**
   ```typescript
   // ❌ Bad
   const testCases = onlyCodeGenerationCases;
   ```

2. **لا تتجاهل التحذيرات**
   ```typescript
   // ❌ Bad
   if (report.isOverfit) {
       // Ignore and deploy anyway
   }
   ```

3. **لا تحسّن على test set**
   ```typescript
   // ❌ Bad
   while (testScore < 0.9) {
       prompt = improvePrompt(prompt, testResults); // Leaking test data!
   }
   ```

4. **لا تستخدم بيانات غير متنوعة**
   ```typescript
   // ❌ Bad
   if (!diversity.isSufficientlyDiverse) {
       // Proceed anyway
   }
   ```

## 🎯 حالات الاستخدام

### 1. تطوير برومبت جديد

```typescript
async function developPrompt() {
    let prompt = initialPrompt;
    const testCases = loadTestCases();
    
    for (let iteration = 0; iteration < 10; iteration++) {
        // Optimize on training data
        const { train, validation } = splitDataset(testCases);
        prompt = await optimizePrompt(prompt, train);
        
        // Check for overfitting
        const trainRes = await test(prompt, train);
        const valRes = await test(prompt, validation);
        const report = await detectOverfitting(prompt, trainRes, valRes);
        
        if (report.isOverfit) {
            console.log(`Iteration ${iteration}: Overfitting detected`);
            prompt = simplifyPrompt(prompt, 0.2);
        } else {
            console.log(`Iteration ${iteration}: Good generalization`);
            break;
        }
    }
    
    return prompt;
}
```

### 2. مقارنة برومبتات متعددة

```typescript
async function compareProm<br/>pts(prompts: string[]) {
    const testCases = loadTestCases();
    const results = [];
    
    for (const prompt of prompts) {
        const analysis = await comprehensiveOverfittingAnalysis(
            prompt,
            testCases,
            executor
        );
        
        results.push({
            prompt,
            isOverfit: analysis.overfittingReport.isOverfit,
            gap: analysis.overfittingReport.gap,
            generalization: analysis.heldOutValidation.generalizationScore,
            stability: analysis.crossValidation.isStable
        });
    }
    
    // اختر الأفضل
    const best = results
        .filter(r => !r.isOverfit)
        .sort((a, b) => b.generalization - a.generalization)[0];
    
    return best.prompt;
}
```

### 3. مراقبة الإنتاج

```typescript
async function monitorProduction() {
    const currentPrompt = loadCurrentPrompt();
    const recentTestCases = loadRecentTestCases(); // آخر 100 استعلام
    
    // تقسيم زمني: 80% قديم (train), 20% جديد (validation)
    const splitDate = Date.now() - 7 * 24 * 60 * 60 * 1000; // آخر أسبوع
    const train = recentTestCases.filter(tc => tc.timestamp < splitDate);
    const validation = recentTestCases.filter(tc => tc.timestamp >= splitDate);
    
    const trainRes = await test(currentPrompt, train);
    const valRes = await test(currentPrompt, validation);
    const report = await detectOverfitting(currentPrompt, trainRes, valRes);
    
    if (report.isOverfit) {
        sendAlert({
            severity: report.severity,
            message: 'Prompt overfitting detected in production',
            recommendation: report.recommendation
        });
    }
}

// تشغيل كل ساعة
setInterval(monitorProduction, 60 * 60 * 1000);
```

## 🔧 التكوين المتقدم

### تخصيص العتبات

```typescript
const customConfig: OverfittingConfig = {
    gapThreshold: 0.05,        // أكثر صرامة
    minAcceptableScore: 0.80,  // معايير أعلى
    maxComplexityTokens: 500,  // برومبتات أقصر
    maxVarianceRatio: 1.5,     // استقرار أعلى
    enableDetailedAnalysis: true
};

const report = await detectOverfitting(
    prompt,
    trainResults,
    valResults,
    customConfig
);
```

### Callbacks مخصصة

```typescript
const analysis = await comprehensiveOverfittingAnalysis(
    prompt,
    testCases,
    executor,
    {
        onProgress: (stage: string, progress: number) => {
            console.log(`${stage}: ${progress}%`);
        },
        onWarning: (warning: string) => {
            logger.warn(warning);
        }
    }
);
```

## 📊 الإحصائيات والتصور

يمكنك استخدام البيانات المستخرجة لإنشاء رسوم بيانية:

```typescript
import { Chart } from 'chart.js';

function visualizeOverfitting(report: OverfittingReport) {
    // رسم بياني للفجوة
    const gapChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Train', 'Validation'],
            datasets: [{
                data: [report.trainScore, report.valScore],
                backgroundColor: ['#4CAF50', '#FFC107']
            }]
        }
    });
    
    // رسم بياني للتباين
    const varianceChart = new Chart(ctx2, {
        type: 'line',
        data: {
            labels: Array.from({ length: 10 }, (_, i) => `Iteration ${i}`),
            datasets: [
                { label: 'Train', data: trainScores },
                { label: 'Val', data: valScores }
            ]
        }
    });
}
```

## 🚀 الخطوات التالية

1. **دمج مع Optimizer**
   ```typescript
   // في hybrid.ts أو genetic.ts
   import { detectOverfitting } from '../evaluation/overfittingDetector';
   
   async function optimize(prompt: string) {
       // ... optimization logic
       
       // Check overfitting after each generation
       const report = await detectOverfitting(prompt, trainRes, valRes);
       if (report.isOverfit) {
           applyRegularization(prompt);
       }
   }
   ```

2. **إضافة إلى Dashboard**
   - عرض Overfitting Status
   - رسوم بيانية للاتجاهات
   - تنبيهات تلقائية

3. **Continuous Monitoring**
   - مراقبة الإنتاج المستمرة
   - A/B Testing
   - Auto-correction

## 📖 المراجع

- [Understanding Overfitting in Machine Learning](https://en.wikipedia.org/wiki/Overfitting)
- [Cross-Validation Techniques](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Regularization Methods](https://en.wikipedia.org/wiki/Regularization_(mathematics))
- [Prompt Engineering Best Practices](https://platform.openai.com/docs/guides/prompt-engineering)

## 📝 الترخيص

Part of Prompt Refiner System - MIT License

---

**تم التنفيذ:** ✅ DIRECTIVE-038 COMPLETE
**التاريخ:** 2025-12-14
**المطور:** AI Coding Agent
