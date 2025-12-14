# RLAIF (Reinforcement Learning from AI Feedback) - DIRECTIVE-035

## نظرة عامة

تم تنفيذ DIRECTIVE-035 بنجاح! هذا النظام يقلل الاعتماد على البشر باستخدام AI للتقييم عبر ثلاث مراحل:

1. **Bootstrap من بيانات بشرية**: تدريب Reward Model على human feedback أولي
2. **Self-Play Loop**: توليد variations وتقييمها بالـ Reward Model وتحسين Policy
3. **Human-in-the-Loop Validation**: مراجعة بشرية دورية لتصحيح أخطاء Reward Model

## المكونات الرئيسية

### `MutationPolicy`
- يختار mutation type بناءً على prompt
- يحسّن نفسه بناءً على rewards المستلمة
- `DefaultMutationPolicy`: تنفيذ بسيط يعتمد على احتمالات متساوية ثم يتحسن

### `RewardModel`
- يتنبأ بجودة prompt variations
- يمكن تدريبه على human feedback
- موجود في `src/models/rewardModel.ts`

### `rlaifTraining()`
- الدالة الرئيسية للتدريب
- تأخذ Policy و Reward Model و config
- ترجع ImprovedPolicy مع إحصائيات التحسن

## الاستخدام

```typescript
import { rlaifTraining, DefaultMutationPolicy } from './training/rlaif';
import { RewardModel } from '../models/rewardModel';

// إنشاء policy و reward model
const policy = new DefaultMutationPolicy();
const rewardModel = new RewardModel();

// تشغيل التدريب
const improvedPolicy = await rlaifTraining(policy, rewardModel, {
  iterations: 10,
  batchSize: 20,
  humanValidationInterval: 3,
  bootstrapFromHumanFeedback: true,
  minHumanFeedbackSamples: 10,
});

// استخدام improved policy
const mutation = improvedPolicy.selectMutation("Write a function...", mutationTypes);
console.log(improvedPolicy.improvementStats);
```

## الإحصائيات المُرجعة

```typescript
{
  startingAverageReward: number;    // متوسط reward في البداية
  endingAverageReward: number;      // متوسط reward في النهاية
  iterations: number;               // عدد iterations المنفذة
  humanValidations: number;         // عدد المراجعات البشرية
  humanCorrections: number;         // عدد التصحيحات البشرية
}
```

## التكامل مع النظام

- **Human Feedback**: يجمع من `src/api/feedback.ts`
- **Training Data**: يجمع من `src/training/dataCollection.ts`
- **Mutations**: يستخدم mutations من `src/mutations.ts`
- **Reward Model**: يستخدم `RewardModel` من `src/models/rewardModel.ts`

## الخطوات التالية (للمرحلة المتقدمة)

1. **تحسين Policy**: يمكن استبدال `DefaultMutationPolicy` بنموذج أكثر تعقيداً (neural network)
2. **تحسين Reward Model**: استخدام embeddings حقيقية بدلاً من features يدوية
3. **UI Integration**: إضافة واجهة للمراجعة البشرية الفعلية
4. **PPO Integration**: استخدام PPO من `src/rl/` للتحسين المتقدم

