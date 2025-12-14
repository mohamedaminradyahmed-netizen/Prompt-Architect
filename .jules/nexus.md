## 2025-12-14 - [Category: 🗃️] Node-safe training data collection
**Insight:** الاعتماد المباشر على `localStorage` يكسر بيئة `testEnvironment: node` ويجعل جمع البيانات غير قابل لإعادة التشغيل خارج المتصفح.
**Action:** أي طبقة Data Collection يجب أن تتحقق من توفر `globalThis.localStorage` قبل القراءة، وتُعيد بيانات فارغة بشكل آمن عند عدم توفره.

## 2025-12-14 - [Category: 🏗️] Type mismatch in model alternatives filtering
**Insight:** `getAlternativeModels` كان يقارن `m.model` (اسم النموذج مثل "gpt-4") مع `this.customModelMap[mode]` (registry key مثل "openai-gpt4")، ما يسبب فشل استبعاد النموذج الحالي من قائمة البدائل.
**Action:** استخدام `Object.entries` بدلاً من `Object.values` لمقارنة registry keys مباشرة، ضامناً عدم ظهور النموذج الحالي في alternatives.

## 2025-12-14 - [Category: ⚡] Division-by-near-zero in monthly savings projection
**Insight:** `estimateMonthlySavings` مع سجل واحد فقط يُنتج `windowMs=1ms`، ما يُضخّم `dailyCalls` إلى 86M ويجعل التوقعات غير واقعية بالمرّة.
**Action:** إضافة early return (`if (this.records.length < 2) return 0`) قبل حساب النافذة الزمنية؛ نافذة واحدة تتطلب على الأقل نقطتين زمنيتين.
