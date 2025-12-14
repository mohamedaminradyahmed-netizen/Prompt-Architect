/**
 * DIRECTIVE-038: معالجة Prompt Overfitting
 * 
 * نظام متكامل للكشف عن overfitting في البرومبتات المُحسّنة
 * والتأكد من أنها تعمل على مدخلات متنوعة.
 * 
 * الاستراتيجيات المطبقة:
 * 1. Diverse Test Sets - اختبار على examples متنوعة
 * 2. Cross-Validation - K-fold validation للـ prompts
 * 3. Held-out Validation - احتفاظ بـ test set منفصل
 * 4. Regularization - معاقبة التعقيد الزائد
 */

import { TestResults, TestCase, LLMExecutor, executeTestSuite } from '../sandbox/testExecutor';

// ==================== التعريفات الأساسية ====================

/**
 * تقرير كشف Overfitting
 */
export interface OverfittingReport {
    /** هل البرومبت overfitted؟ */
    isOverfit: boolean;

    /** متوسط النقاط على بيانات التدريب */
    trainScore: number;

    /** متوسط النقاط على بيانات التحقق */
    valScore: number;

    /** الفجوة بين Train و Validation (trainScore - valScore) */
    gap: number;

    /** درجة الثقة في التصنيف (0-1) */
    confidence: number;

    /** شدة الـ overfitting (0-1، أعلى = أسوأ) */
    severity: 'none' | 'mild' | 'moderate' | 'severe';

    /** التوصية لمعالجة المشكلة */
    recommendation: string;

    /** تحليل تفصيلي */
    analysis: OverfittingAnalysis;
}

/**
 * تحليل تفصيلي لـ Overfitting
 */
export interface OverfittingAnalysis {
    /** تحليل التباين في الأداء */
    varianceAnalysis: {
        trainVariance: number;
        valVariance: number;
        varianceRatio: number; // valVariance / trainVariance
    };

    /** تحليل التعقيد */
    complexityAnalysis: {
        promptLength: number;
        tokenCount: number;
        complexityScore: number;
        isOverlyComplex: boolean;
    };

    /** تحليل الأداء حسب الفئة */
    categoryPerformance?: Map<string, {
        trainScore: number;
        valScore: number;
        gap: number;
    }>;

    /** نقاط الفشل الرئيسية */
    failurePoints: string[];
}

/**
 * إعدادات كشف Overfitting
 */
export interface OverfittingConfig {
    /** عتبة الفجوة المقبولة (default: 0.1 = 10%) */
    gapThreshold?: number;

    /** الحد الأدنى للنقاط المطلوبة (default: 0.7) */
    minAcceptableScore?: number;

    /** عتبة التعقيد الزائد (default: 1000 tokens) */
    maxComplexityTokens?: number;

    /** نسبة التباين المقبولة (default: 2.0) */
    maxVarianceRatio?: number;

    /** تفعيل التحليل المفصل */
    enableDetailedAnalysis?: boolean;
}

/**
 * نتيجة K-Fold Cross Validation
 */
export interface CrossValidationResult {
    /** عدد الـ folds */
    folds: number;

    /** نقاط كل fold */
    foldScores: number[];

    /** متوسط النقاط */
    meanScore: number;

    /** الانحراف المعياري */
    stdDeviation: number;

    /** أفضل fold */
    bestFold: number;

    /** أسوأ fold */
    worstFold: number;

    /** هل النتائج مستقرة؟ */
    isStable: boolean;
}

/**
 * نتيجة تحليل مجموعة بيانات محتفظ بها
 */
export interface HeldOutValidationResult {
    /** نقاط Training Set */
    trainScore: number;

    /** نقاط Validation Set */
    valScore: number;

    /** نقاط Test Set (held-out) */
    testScore: number;

    /** الفجوة بين Train و Test */
    trainTestGap: number;

    /** تعميم النموذج (generalization) */
    generalizationScore: number;
}

/**
 * نتيجة تحليل التنوع
 */
export interface DiversityAnalysis {
    /** معامل التنوع (0-1، أعلى = أكثر تنوعاً) */
    diversityScore: number;

    /** عدد الفئات المختلفة */
    uniqueCategories: number;

    /** توزيع الأداء حسب الفئة */
    categoryDistribution: Map<string, number>;

    /** هل مجموعة البيانات متنوعة بشكل كافٍ؟ */
    isSufficientlyDiverse: boolean;
}

// ==================== الوظيفة الرئيسية ====================

/**
 * كشف Overfitting في البرومبت
 * 
 * @param prompt البرومبت المُحسّن
 * @param trainResults نتائج الأداء على بيانات التدريب
 * @param valResults نتائج الأداء على بيانات التحقق
 * @param config إعدادات الكشف (اختياري)
 * @returns تقرير شامل عن حالة Overfitting
 */
export async function detectOverfitting(
    prompt: string,
    trainResults: TestResults,
    valResults: TestResults,
    config?: OverfittingConfig
): Promise<OverfittingReport> {

    // الإعدادات الافتراضية
    const cfg: Required<OverfittingConfig> = {
        gapThreshold: config?.gapThreshold ?? 0.10,
        minAcceptableScore: config?.minAcceptableScore ?? 0.70,
        maxComplexityTokens: config?.maxComplexityTokens ?? 1000,
        maxVarianceRatio: config?.maxVarianceRatio ?? 2.0,
        enableDetailedAnalysis: config?.enableDetailedAnalysis ?? true
    };

    // استخراج النقاط
    const trainScore = trainResults.aggregateScore;
    const valScore = valResults.aggregateScore;
    const gap = trainScore - valScore;

    // حساب التباين
    const trainVariance = calculateVariance(trainResults.results.map(r => r.score));
    const valVariance = calculateVariance(valResults.results.map(r => r.score));
    const varianceRatio = trainVariance > 0 ? valVariance / trainVariance : 1.0;

    // تحليل التعقيد
    const complexityAnalysis = analyzeComplexity(prompt, cfg.maxComplexityTokens);

    // تحديد حالة Overfitting
    const isOverfit = gap > cfg.gapThreshold ||
        varianceRatio > cfg.maxVarianceRatio ||
        (trainScore > cfg.minAcceptableScore && valScore < cfg.minAcceptableScore);

    // تحديد الشدة
    let severity: 'none' | 'mild' | 'moderate' | 'severe' = 'none';
    if (isOverfit) {
        if (gap > 0.30 || varianceRatio > 4.0) severity = 'severe';
        else if (gap > 0.20 || varianceRatio > 3.0) severity = 'moderate';
        else severity = 'mild';
    }

    // حساب الثقة
    const confidence = calculateConfidence(
        trainResults.results.length,
        valResults.results.length,
        varianceRatio
    );

    // تحليل نقاط الفشل
    const failurePoints = identifyFailurePoints(trainResults, valResults);

    // توليد التوصية
    const recommendation = generateRecommendation(
        isOverfit,
        severity,
        gap,
        complexityAnalysis,
        varianceRatio
    );

    // التقرير النهائي
    return {
        isOverfit,
        trainScore,
        valScore,
        gap,
        confidence,
        severity,
        recommendation,
        analysis: {
            varianceAnalysis: {
                trainVariance,
                valVariance,
                varianceRatio
            },
            complexityAnalysis,
            failurePoints
        }
    };
}

// ==================== K-Fold Cross Validation ====================

/**
 * K-Fold Cross Validation للبرومبتات
 * 
 * يقسم البيانات إلى K أجزاء ويختبر على كل جزء
 * لقياس استقرار الأداء
 * 
 * @param prompt البرومبت للاختبار
 * @param testCases جميع حالات الاختبار
 * @param executor وظيفة تنفيذ LLM
 * @param k عدد الـ folds (default: 5)
 * @returns نتيجة Cross Validation
 */
export async function kFoldCrossValidation(
    prompt: string,
    testCases: TestCase[],
    executor: LLMExecutor,
    k: number = 5
): Promise<CrossValidationResult> {

    if (k < 2) throw new Error('K must be at least 2');
    if (testCases.length < k) throw new Error(`Not enough test cases (${testCases.length}) for ${k}-fold validation`);

    // خلط البيانات
    const shuffled = [...testCases].sort(() => Math.random() - 0.5);
    const foldSize = Math.floor(shuffled.length / k);

    const foldScores: number[] = [];

    // تشغيل كل fold
    for (let i = 0; i < k; i++) {
        // تقسيم البيانات
        const start = i * foldSize;
        const end = i === k - 1 ? shuffled.length : start + foldSize;

        const testFold = shuffled.slice(start, end);

        // تنفيذ الاختبار
        const results = await executeTestSuite([prompt], testFold, executor);
        foldScores.push(results[0].aggregateScore);
    }

    // حساب الإحصائيات
    const meanScore = foldScores.reduce((a, b) => a + b, 0) / foldScores.length;
    const stdDeviation = Math.sqrt(
        foldScores.reduce((sum, score) => sum + Math.pow(score - meanScore, 2), 0) / foldScores.length
    );

    const bestFold = foldScores.indexOf(Math.max(...foldScores));
    const worstFold = foldScores.indexOf(Math.min(...foldScores));

    // النتائج مستقرة إذا كان الانحراف المعياري صغيراً
    const isStable = stdDeviation < 0.15; // أقل من 15%

    return {
        folds: k,
        foldScores,
        meanScore,
        stdDeviation,
        bestFold,
        worstFold,
        isStable
    };
}

// ==================== Held-out Validation ====================

/**
 * تقسيم البيانات إلى Train/Val/Test
 * 
 * @param testCases جميع حالات الاختبار
 * @param trainRatio نسبة التدريب (default: 0.6)
 * @param valRatio نسبة التحقق (default: 0.2)
 * @returns البيانات المقسمة
 */
export function splitDataset(
    testCases: TestCase[],
    trainRatio: number = 0.6,
    valRatio: number = 0.2
): {
    train: TestCase[];
    validation: TestCase[];
    test: TestCase[];
} {
    const shuffled = [...testCases].sort(() => Math.random() - 0.5);

    const trainSize = Math.floor(shuffled.length * trainRatio);
    const valSize = Math.floor(shuffled.length * valRatio);

    return {
        train: shuffled.slice(0, trainSize),
        validation: shuffled.slice(trainSize, trainSize + valSize),
        test: shuffled.slice(trainSize + valSize)
    };
}

/**
 * تحقق Held-out كامل
 * 
 * @param prompt البرومبت للاختبار
 * @param testCases جميع حالات الاختبار
 * @param executor وظيفة تنفيذ LLM
 * @returns نتيجة التحقق الكامل
 */
export async function heldOutValidation(
    prompt: string,
    testCases: TestCase[],
    executor: LLMExecutor,
    splits?: { train: TestCase[]; validation: TestCase[]; test: TestCase[] }
): Promise<HeldOutValidationResult> {

    const { train, validation, test } = splits || splitDataset(testCases);

    // تشغيل على كل مجموعة
    const [trainResults, valResults, testResults] = await Promise.all([
        executeTestSuite([prompt], train, executor),
        executeTestSuite([prompt], validation, executor),
        executeTestSuite([prompt], test, executor)
    ]);

    const trainScore = trainResults[0].aggregateScore;
    const valScore = valResults[0].aggregateScore;
    const testScore = testResults[0].aggregateScore;

    const trainTestGap = trainScore - testScore;

    // Generalization Score: مدى قرب أداء Test من Train
    const generalizationScore = Math.max(0, 1 - Math.abs(trainTestGap));

    return {
        trainScore,
        valScore,
        testScore,
        trainTestGap,
        generalizationScore
    };
}

// ==================== Diversity Analysis ====================

/**
 * تحليل تنوع مجموعة البيانات
 * 
 * @param testCases حالات الاختبار
 * @returns تحليل التنوع
 */
export function analyzeDiversity(testCases: TestCase[]): DiversityAnalysis {
    // استخراج الفئات من metadata
    const categories = new Map<string, number>();

    testCases.forEach(tc => {
        const category = tc.metadata?.category || 'uncategorized';
        categories.set(category, (categories.get(category) || 0) + 1);
    });

    const uniqueCategories = categories.size;

    // حساب معامل التنوع (Shannon Entropy)
    const total = testCases.length;
    let entropy = 0;

    categories.forEach(count => {
        const p = count / total;
        entropy -= p * Math.log2(p);
    });

    // تطبيع Entropy (0-1)
    const maxEntropy = Math.log2(uniqueCategories || 1);
    const diversityScore = maxEntropy > 0 ? entropy / maxEntropy : 0;

    // متنوع بشكل كافٍ إذا كان هناك على الأقل 3 فئات وDiversity > 0.6
    const isSufficientlyDiverse = uniqueCategories >= 3 && diversityScore > 0.6;

    return {
        diversityScore,
        uniqueCategories,
        categoryDistribution: categories,
        isSufficientlyDiverse
    };
}

// ==================== Regularization ====================

/**
 * حساب معامل Regularization للبرومبت
 * 
 * يعاقب البرومبتات المعقدة بشكل مفرط
 * 
 * @param prompt البرومبت
 * @param lambda معامل Regularization (default: 0.001)
 * @returns قيمة العقوبة
 */
export function calculateRegularization(prompt: string, lambda: number = 0.001): number {
    // L1 Regularization: معاقبة الطول
    const l1Penalty = prompt.length * lambda;

    // L2 Regularization: معاقبة التعقيد
    const tokenCount = estimateTokenCount(prompt);
    const l2Penalty = Math.pow(tokenCount, 2) * lambda;

    // العقوبة الكلية
    return l1Penalty + l2Penalty;
}

/**
 * تبسيط البرومبت لتقليل Overfitting
 * 
 * @param prompt البرومبت الأصلي
 * @param targetReduction نسبة التقليل المستهدفة (0-1)
 * @returns البرومبت المبسط
 */
export function simplifyPrompt(prompt: string, targetReduction: number = 0.3): string {
    const lines = prompt.split('\n');
    const targetLines = Math.ceil(lines.length * (1 - targetReduction));

    // استراتيجيات التبسيط:

    // 1. إزالة الأمثلة الطويلة
    const withoutExamples = lines.filter(line => {
        const isExample = line.toLowerCase().includes('example:') ||
            line.toLowerCase().includes('e.g.') ||
            line.toLowerCase().includes('for instance');
        return !isExample || line.length < 100;
    });

    // 2. إزالة الشروح الزائدة
    const withoutExplanations = withoutExamples.filter(line => {
        const isExplanation = line.toLowerCase().includes('note:') ||
            line.toLowerCase().includes('explanation:') ||
            line.toLowerCase().includes('in other words');
        return !isExplanation;
    });

    // 3. دمج التعليمات المتكررة
    const unique = Array.from(new Set(withoutExplanations));

    // 4. الاحتفاظ بالخطوط الأكثر أهمية
    const important = unique.slice(0, Math.max(targetLines, 5));

    return important.join('\n').trim();
}

// ==================== الوظائف المساعدة ====================

/**
 * حساب التباين
 */
function calculateVariance(scores: number[]): number {
    if (scores.length === 0) return 0;

    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    const squaredDiffs = scores.map(score => Math.pow(score - mean, 2));
    return squaredDiffs.reduce((a, b) => a + b, 0) / scores.length;
}

/**
 * تحليل تعقيد البرومبت
 */
function analyzeComplexity(prompt: string, maxTokens: number): {
    promptLength: number;
    tokenCount: number;
    complexityScore: number;
    isOverlyComplex: boolean;
} {
    const promptLength = prompt.length;
    const tokenCount = estimateTokenCount(prompt);

    // حساب معامل التعقيد (0-1)
    // يأخذ بعين الاعتبار: الطول، عدد الأسطر، الكلمات المعقدة
    const lines = prompt.split('\n').length;
    const avgLineLength = promptLength / lines;
    const longWords = prompt.split(/\s+/).filter(w => w.length > 10).length;

    const complexityScore = Math.min(1,
        (tokenCount / maxTokens) * 0.5 +
        (avgLineLength / 100) * 0.3 +
        (longWords / 20) * 0.2
    );

    const isOverlyComplex = tokenCount > maxTokens || complexityScore > 0.7;

    return {
        promptLength,
        tokenCount,
        complexityScore,
        isOverlyComplex
    };
}

/**
 * تقدير عدد التوكنات (تقريبي)
 */
function estimateTokenCount(text: string): number {
    // تقدير بسيط: ~4 أحرف = 1 token
    return Math.ceil(text.length / 4);
}

/**
 * حساب معامل الثقة في التصنيف
 */
function calculateConfidence(
    trainSize: number,
    valSize: number,
    varianceRatio: number
): number {
    // الثقة تزداد مع:
    // 1. حجم البيانات الأكبر
    // 2. نسبة تباين معقولة

    const sizeConfidence = Math.min(1, (trainSize + valSize) / 100);
    const varianceConfidence = Math.max(0, 1 - Math.abs(varianceRatio - 1) / 2);

    return (sizeConfidence + varianceConfidence) / 2;
}

/**
 * تحديد نقاط الفشل الرئيسية
 */
function identifyFailurePoints(
    trainResults: TestResults,
    valResults: TestResults
): string[] {
    const failures: string[] = [];

    // 1. مقارنة معدلات النجاح
    const passRateDiff = trainResults.passRate - valResults.passRate;
    if (passRateDiff > 0.2) {
        failures.push(`معدل النجاح انخفض بنسبة ${(passRateDiff * 100).toFixed(1)}% في التحقق`);
    }

    // 2. فحص الاختبارات التي نجحت في التدريب وفشلت في التحقق
    const trainPassed = new Set(
        trainResults.results.filter(r => r.passed).map(r => r.testCaseId)
    );
    const valFailed = valResults.results.filter(r => !r.passed);

    valFailed.forEach(result => {
        if (trainPassed.has(result.testCaseId)) {
            failures.push(`الاختبار ${result.testCaseId} نجح في التدريب لكن فشل في التحقق`);
        }
    });

    // 3. تحليل التباين الكبير في النقاط
    const scoreVariances = valResults.results.map((valRes, idx) => {
        const trainRes = trainResults.results[idx];
        if (!trainRes) return 0;
        return Math.abs(trainRes.score - valRes.score);
    });

    const highVarianceCount = scoreVariances.filter(v => v > 0.3).length;
    if (highVarianceCount > scoreVariances.length * 0.3) {
        failures.push(`${highVarianceCount} اختبار يظهر تبايناً كبيراً في الأداء`);
    }

    return failures.slice(0, 5); // أهم 5 نقاط
}

/**
 * توليد توصية لمعالجة Overfitting
 */
function generateRecommendation(
    isOverfit: boolean,
    severity: 'none' | 'mild' | 'moderate' | 'severe',
    gap: number,
    complexity: ReturnType<typeof analyzeComplexity>,
    varianceRatio: number
): string {
    if (!isOverfit) {
        return '✅ البرومبت يعمل بشكل جيد على بيانات متنوعة. لا حاجة لتعديلات.';
    }

    const recommendations: string[] = [];

    // توصيات حسب الشدة
    if (severity === 'severe') {
        recommendations.push('🚨 Overfitting حاد: إعادة تصميم البرومبت مطلوبة');
    } else if (severity === 'moderate') {
        recommendations.push('⚠️ Overfitting متوسط: تعديلات مهمة مطلوبة');
    } else {
        recommendations.push('⚡ Overfitting طفيف: تحسينات بسيطة مطلوبة');
    }

    // توصيات حسب الفجوة
    if (gap > 0.15) {
        recommendations.push(`• قلل تخصيص البرومبت لبيانات التدريب (الفجوة: ${(gap * 100).toFixed(1)}%)`);
        recommendations.push('• أضف المزيد من البيانات المتنوعة للتدريب');
    }

    // توصيات حسب التعقيد
    if (complexity.isOverlyComplex) {
        recommendations.push(`• بسّط البرومبت (حالياً ${complexity.tokenCount} tokens)`);
        recommendations.push('• أزل الأمثلة أو الشروح الزائدة');
        recommendations.push(`• استهدف تقليل 30-40% من الطول`);
    }

    // توصيات حسب التباين
    if (varianceRatio > 2.0) {
        recommendations.push(`• النتائج غير مستقرة (نسبة التباين: ${varianceRatio.toFixed(2)})`);
        recommendations.push('• أضف قيوداً أكثر وضوحاً في البرومبت');
        recommendations.push('• استخدم أمثلة متسقة');
    }

    // توصيات عامة
    recommendations.push('• استخدم Cross-Validation للتحقق من الاستقرار');
    recommendations.push('• احتفظ بـ test set منفصل للتقييم النهائي');

    return recommendations.join('\n');
}

// ==================== تصدير مُجمّع ====================

/**
 * نظام كامل لتقييم Overfitting
 * 
 * يشغل جميع الاستراتيجيات ويعطي تقرير شامل
 * 
 * @param prompt البرومبت المُحسّن
 * @param testCases جميع حالات الاختبار
 * @param executor وظيفة تنفيذ LLM
 * @param config إعدادات التقييم
 * @returns تقرير شامل
 */
export async function comprehensiveOverfittingAnalysis(
    prompt: string,
    testCases: TestCase[],
    executor: LLMExecutor,
    config?: OverfittingConfig
): Promise<{
    overfittingReport: OverfittingReport;
    crossValidation: CrossValidationResult;
    heldOutValidation: HeldOutValidationResult;
    diversityAnalysis: DiversityAnalysis;
    regularizationPenalty: number;
    simplifiedPrompt?: string;
}> {

    // 1. تحليل التنوع
    const diversityAnalysis = analyzeDiversity(testCases);

    if (!diversityAnalysis.isSufficientlyDiverse) {
        console.warn('⚠️ تحذير: مجموعة البيانات غير متنوعة بشكل كافٍ. النتائج قد لا تكون موثوقة.');
    }

    // 2. تقسيم البيانات
    const { train, validation, test } = splitDataset(testCases);

    // 3. تشغيل على Train و Validation
    const [trainResults, valResults] = await Promise.all([
        executeTestSuite([prompt], train, executor),
        executeTestSuite([prompt], validation, executor)
    ]);

    // 4. كشف Overfitting
    const overfittingReport = await detectOverfitting(
        prompt,
        trainResults[0],
        valResults[0],
        config
    );

    // 5. Cross Validation
    const crossValidation = await kFoldCrossValidation(
        prompt,
        [...train, ...validation],
        executor,
        5
    );

    // 6. Held-out Validation
    // NOTE(Why): تجنب shadowing/TDZ. اسم المتغير لا يجب أن يطابق اسم الدالة المستوردة/المعرفة.
    const heldOutResult = await heldOutValidation(prompt, testCases, executor, { train, validation, test });

    // 7. حساب Regularization
    const regularizationPenalty = calculateRegularization(prompt);

    // 8. توليد نسخة مبسطة إذا لزم الأمر
    let simplifiedPrompt: string | undefined;
    if (overfittingReport.analysis.complexityAnalysis.isOverlyComplex) {
        simplifiedPrompt = simplifyPrompt(prompt, 0.3);
    }

    return {
        overfittingReport,
        crossValidation,
        heldOutValidation: heldOutResult,
        diversityAnalysis,
        regularizationPenalty,
        simplifiedPrompt
    };
}

/**
 * دالة مساعدة لطباعة تقرير مفصل
 */
export function printOverfittingReport(report: OverfittingReport): void {
    console.log('\n' + '='.repeat(60));
    console.log('📊 تقرير كشف Overfitting');
    console.log('='.repeat(60));

    console.log(`\n🎯 النتيجة: ${report.isOverfit ? '⚠️ OVERFITTED' : '✅ GOOD'}`);
    console.log(`📈 الشدة: ${report.severity.toUpperCase()}`);
    console.log(`🎲 الثقة: ${(report.confidence * 100).toFixed(1)}%`);

    console.log('\n📊 النقاط:');
    console.log(`  • Training Score:   ${(report.trainScore * 100).toFixed(1)}%`);
    console.log(`  • Validation Score: ${(report.valScore * 100).toFixed(1)}%`);
    console.log(`  • Gap:              ${(report.gap * 100).toFixed(1)}%`);

    console.log('\n🔍 تحليل التباين:');
    const variance = report.analysis.varianceAnalysis;
    console.log(`  • Train Variance:   ${variance.trainVariance.toFixed(4)}`);
    console.log(`  • Val Variance:     ${variance.valVariance.toFixed(4)}`);
    console.log(`  • Variance Ratio:   ${variance.varianceRatio.toFixed(2)}`);

    console.log('\n🧩 تحليل التعقيد:');
    const complexity = report.analysis.complexityAnalysis;
    console.log(`  • Prompt Length:    ${complexity.promptLength} chars`);
    console.log(`  • Token Count:      ${complexity.tokenCount} tokens`);
    console.log(`  • Complexity Score: ${(complexity.complexityScore * 100).toFixed(1)}%`);
    console.log(`  • Too Complex:      ${complexity.isOverlyComplex ? 'YES ⚠️' : 'NO ✅'}`);

    if (report.analysis.failurePoints.length > 0) {
        console.log('\n❌ نقاط الفشل:');
        report.analysis.failurePoints.forEach(fp => console.log(`  • ${fp}`));
    }

    console.log('\n💡 التوصية:');
    console.log(report.recommendation.split('\n').map(line => `  ${line}`).join('\n'));

    console.log('\n' + '='.repeat(60) + '\n');
}
