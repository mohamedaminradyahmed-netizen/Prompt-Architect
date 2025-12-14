/**
 * DEMO: Overfitting Detection System
 * 
 * أمثلة عملية لاستخدام نظام كشف Overfitting
 */

import {
    detectOverfitting,
    kFoldCrossValidation,
    heldOutValidation,
    analyzeDiversity,
    calculateRegularization,
    simplifyPrompt,
    comprehensiveOverfittingAnalysis,
    printOverfittingReport,
    OverfittingConfig
} from './overfittingDetector';
import { TestCase, LLMExecutor } from '../sandbox/testExecutor';

// ==================== Mock LLM Executor ====================

/**
 * Mock LLM Executor للتجربة
 * يحاكي سلوك LLM حقيقي مع overfitting
 */
const createMockExecutor = (overfitDegree: number = 0): LLMExecutor => {
    return async (prompt: string): Promise<string> => {
        // محاكاة latency
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // محاكاة overfitting: الأداء أفضل على prompts طويلة (training)
        // وأسوأ على prompts قصيرة (validation)
        const isTrainingLike = prompt.length > 500;
        const baseQuality = 0.8;
        
        if (isTrainingLike) {
            // أداء جيد على بيانات التدريب
            return Math.random() > (0.1 - overfitDegree * 0.1) 
                ? 'SUCCESS: High quality response' 
                : 'PARTIAL: Good but not perfect';
        } else {
            // أداء أسوأ على بيانات جديدة
            return Math.random() > (0.3 + overfitDegree * 0.2) 
                ? 'SUCCESS: Adequate response' 
                : 'FAILURE: Poor quality';
        }
    };
};

// ==================== Sample Test Cases ====================

const createSampleTestCases = (): TestCase[] => {
    return [
        // Code generation tasks
        {
            id: 'code-1',
            prompt: 'Write a function to sort an array',
            evaluationCriteria: {
                matchType: 'includes',
                matchValue: 'SUCCESS'
            },
            metadata: { category: 'code_generation' }
        },
        {
            id: 'code-2',
            prompt: 'Create a binary search implementation',
            evaluationCriteria: {
                matchType: 'includes',
                matchValue: 'SUCCESS'
            },
            metadata: { category: 'code_generation' }
        },
        {
            id: 'code-3',
            prompt: 'Implement a linked list',
            evaluationCriteria: {
                matchType: 'includes',
                matchValue: 'SUCCESS'
            },
            metadata: { category: 'code_generation' }
        },
        
        // Content writing tasks
        {
            id: 'content-1',
            prompt: 'Write a blog post about AI',
            evaluationCriteria: {
                matchType: 'includes',
                matchValue: 'SUCCESS'
            },
            metadata: { category: 'content_writing' }
        },
        {
            id: 'content-2',
            prompt: 'Create a product description',
            evaluationCriteria: {
                matchType: 'includes',
                matchValue: 'SUCCESS'
            },
            metadata: { category: 'content_writing' }
        },
        {
            id: 'content-3',
            prompt: 'Draft an email template',
            evaluationCriteria: {
                matchType: 'includes',
                matchValue: 'SUCCESS'
            },
            metadata: { category: 'content_writing' }
        },
        
        // Data analysis tasks
        {
            id: 'analysis-1',
            prompt: 'Analyze sales trends',
            evaluationCriteria: {
                matchType: 'includes',
                matchValue: 'SUCCESS'
            },
            metadata: { category: 'data_analysis' }
        },
        {
            id: 'analysis-2',
            prompt: 'Summarize quarterly report',
            evaluationCriteria: {
                matchType: 'includes',
                matchValue: 'SUCCESS'
            },
            metadata: { category: 'data_analysis' }
        },
        
        // Marketing tasks
        {
            id: 'marketing-1',
            prompt: 'Create ad copy for product launch',
            evaluationCriteria: {
                matchType: 'includes',
                matchValue: 'SUCCESS'
            },
            metadata: { category: 'marketing' }
        },
        {
            id: 'marketing-2',
            prompt: 'Write social media posts',
            evaluationCriteria: {
                matchType: 'includes',
                matchValue: 'SUCCESS'
            },
            metadata: { category: 'marketing' }
        },
        
        // Fill to reach 20+ test cases
        ...Array.from({ length: 10 }, (_, i) => ({
            id: `test-${i + 11}`,
            prompt: `Generic task ${i + 11}`,
            evaluationCriteria: {
                matchType: 'includes' as const,
                matchValue: 'SUCCESS'
            },
            metadata: { category: i % 2 === 0 ? 'code_generation' : 'content_writing' }
        }))
    ];
};

// ==================== DEMO 1: Basic Overfitting Detection ====================

export async function demo1_basicDetection() {
    console.log('\n' + '='.repeat(70));
    console.log('🎯 DEMO 1: Basic Overfitting Detection');
    console.log('='.repeat(70));
    
    const testCases = createSampleTestCases();
    const executor = createMockExecutor(0.5); // Moderate overfitting
    
    // برومبت معقد (محتمل overfitting)
    const complexPrompt = `
You are an expert code generator with 15 years of experience.
When writing code, always follow these rules:
1. Use descriptive variable names
2. Add comprehensive comments
3. Include error handling for all edge cases
4. Write unit tests for each function
5. Follow the single responsibility principle
6. Use design patterns where appropriate
7. Optimize for both readability and performance
8. Consider memory efficiency
9. Handle null/undefined cases
10. Add logging for debugging

Examples:
- For sorting: Use merge sort for large arrays, quicksort for small ones
- For searching: Binary search when sorted, linear when unsorted
- For data structures: Choose based on time/space complexity requirements

Additional context:
- Target ES2020+ syntax
- TypeScript preferred
- Functional programming style when possible
- Immutability by default

Remember to consider all edge cases and provide complete, production-ready code.
`;
    
    // تقسيم البيانات
    const trainSize = Math.floor(testCases.length * 0.6);
    const trainCases = testCases.slice(0, trainSize);
    const valCases = testCases.slice(trainSize);
    
    // تشغيل الاختبارات
    const { executeTestSuite } = await import('../sandbox/testExecutor');
    const [trainResults, valResults] = await Promise.all([
        executeTestSuite([complexPrompt], trainCases, executor),
        executeTestSuite([complexPrompt], valCases, executor)
    ]);
    
    // كشف Overfitting
    const report = await detectOverfitting(
        complexPrompt,
        trainResults[0],
        valResults[0]
    );
    
    printOverfittingReport(report);
    
    // عرض البرومبت المبسط إذا كان معقداً
    if (report.analysis.complexityAnalysis.isOverlyComplex) {
        console.log('📝 البرومبت المبسط المقترح:');
        console.log('─'.repeat(70));
        const simplified = simplifyPrompt(complexPrompt, 0.4);
        console.log(simplified);
        console.log('─'.repeat(70));
    }
}

// ==================== DEMO 2: K-Fold Cross Validation ====================

export async function demo2_crossValidation() {
    console.log('\n' + '='.repeat(70));
    console.log('🔄 DEMO 2: K-Fold Cross Validation');
    console.log('='.repeat(70));
    
    const testCases = createSampleTestCases();
    const executor = createMockExecutor(0.3);
    
    const prompt = `
You are a helpful AI assistant.
Provide clear, accurate, and concise responses.
Focus on solving the user's problem effectively.
`;
    
    console.log('\n🚀 بدء 5-Fold Cross Validation...\n');
    
    const cvResult = await kFoldCrossValidation(prompt, testCases, executor, 5);
    
    console.log('📊 نتائج Cross Validation:');
    console.log('─'.repeat(70));
    console.log(`عدد الـ Folds:          ${cvResult.folds}`);
    console.log(`متوسط النقاط:          ${(cvResult.meanScore * 100).toFixed(1)}%`);
    console.log(`الانحراف المعياري:     ${(cvResult.stdDeviation * 100).toFixed(1)}%`);
    console.log(`أفضل Fold:             #${cvResult.bestFold + 1} (${(cvResult.foldScores[cvResult.bestFold] * 100).toFixed(1)}%)`);
    console.log(`أسوأ Fold:             #${cvResult.worstFold + 1} (${(cvResult.foldScores[cvResult.worstFold] * 100).toFixed(1)}%)`);
    console.log(`الاستقرار:             ${cvResult.isStable ? '✅ مستقر' : '⚠️ غير مستقر'}`);
    
    console.log('\n📈 نقاط كل Fold:');
    cvResult.foldScores.forEach((score, idx) => {
        const bar = '█'.repeat(Math.round(score * 50));
        console.log(`  Fold ${idx + 1}: ${bar} ${(score * 100).toFixed(1)}%`);
    });
    
    console.log('\n💡 التفسير:');
    if (cvResult.isStable) {
        console.log('  ✅ النتائج مستقرة عبر جميع الـ folds');
        console.log('  ✅ البرومبت يعمل بشكل متسق على بيانات مختلفة');
    } else {
        console.log('  ⚠️ النتائج تختلف بشكل كبير بين الـ folds');
        console.log('  ⚠️ قد يكون هناك overfitting أو البيانات غير متجانسة');
    }
}

// ==================== DEMO 3: Held-out Validation ====================

export async function demo3_heldOutValidation() {
    console.log('\n' + '='.repeat(70));
    console.log('🎯 DEMO 3: Held-out Validation (Train/Val/Test Split)');
    console.log('='.repeat(70));
    
    const testCases = createSampleTestCases();
    const executor = createMockExecutor(0.4);
    
    const prompt = `
Generate high-quality responses that are:
- Accurate and factual
- Clear and well-structured
- Appropriate for the context
- Professional in tone
`;
    
    console.log('\n🔍 تشغيل Held-out Validation...\n');
    
    const result = await heldOutValidation(prompt, testCases, executor);
    
    console.log('📊 النتائج:');
    console.log('─'.repeat(70));
    console.log(`Training Score:        ${(result.trainScore * 100).toFixed(1)}%`);
    console.log(`Validation Score:      ${(result.valScore * 100).toFixed(1)}%`);
    console.log(`Test Score:            ${(result.testScore * 100).toFixed(1)}%`);
    console.log(`Train-Test Gap:        ${(result.trainTestGap * 100).toFixed(1)}%`);
    console.log(`Generalization Score:  ${(result.generalizationScore * 100).toFixed(1)}%`);
    
    console.log('\n📈 التصور البياني:');
    const trainBar = '█'.repeat(Math.round(result.trainScore * 50));
    const valBar = '█'.repeat(Math.round(result.valScore * 50));
    const testBar = '█'.repeat(Math.round(result.testScore * 50));
    
    console.log(`  Train: ${trainBar} ${(result.trainScore * 100).toFixed(1)}%`);
    console.log(`  Val:   ${valBar} ${(result.valScore * 100).toFixed(1)}%`);
    console.log(`  Test:  ${testBar} ${(result.testScore * 100).toFixed(1)}%`);
    
    console.log('\n💡 التقييم:');
    if (result.trainTestGap < 0.1) {
        console.log('  ✅ تعميم ممتاز! الأداء متسق عبر جميع المجموعات');
    } else if (result.trainTestGap < 0.2) {
        console.log('  ⚠️ تعميم جيد، لكن هناك مجال للتحسين');
    } else {
        console.log('  ❌ overfitting واضح! الأداء ينخفض بشكل كبير على بيانات جديدة');
    }
}

// ==================== DEMO 4: Diversity Analysis ====================

export async function demo4_diversityAnalysis() {
    console.log('\n' + '='.repeat(70));
    console.log('🌈 DEMO 4: Dataset Diversity Analysis');
    console.log('='.repeat(70));
    
    const testCases = createSampleTestCases();
    
    const diversity = analyzeDiversity(testCases);
    
    console.log('\n📊 تحليل التنوع:');
    console.log('─'.repeat(70));
    console.log(`معامل التنوع:         ${(diversity.diversityScore * 100).toFixed(1)}%`);
    console.log(`عدد الفئات الفريدة:   ${diversity.uniqueCategories}`);
    console.log(`تنوع كافٍ:           ${diversity.isSufficientlyDiverse ? '✅ نعم' : '❌ لا'}`);
    
    console.log('\n📈 توزيع الفئات:');
    const total = testCases.length;
    Array.from(diversity.categoryDistribution.entries())
        .sort((a, b) => b[1] - a[1])
        .forEach(([category, count]) => {
            const percentage = (count / total * 100).toFixed(1);
            const bar = '█'.repeat(Math.round(count / total * 50));
            console.log(`  ${category.padEnd(20)} ${bar} ${count} (${percentage}%)`);
        });
    
    console.log('\n💡 التوصيات:');
    if (diversity.isSufficientlyDiverse) {
        console.log('  ✅ مجموعة البيانات متنوعة بشكل جيد');
        console.log('  ✅ يمكن الاعتماد على نتائج التقييم');
    } else {
        console.log('  ⚠️ مجموعة البيانات تفتقر للتنوع');
        console.log('  ⚠️ أضف المزيد من الفئات المختلفة');
        console.log('  ⚠️ نتائج التقييم قد لا تكون ممثلة');
    }
}

// ==================== DEMO 5: Regularization ====================

export async function demo5_regularization() {
    console.log('\n' + '='.repeat(70));
    console.log('⚖️ DEMO 5: Prompt Regularization & Simplification');
    console.log('='.repeat(70));
    
    const prompts = [
        {
            name: 'Simple Prompt',
            text: 'Generate a Python function to sort a list.'
        },
        {
            name: 'Moderate Prompt',
            text: `
Generate a Python function to sort a list.
Use efficient algorithms and add error handling.
Include docstrings and type hints.
`.trim()
        },
        {
            name: 'Complex Prompt',
            text: `
You are an expert Python developer with deep knowledge of algorithms.

Task: Generate a sorting function with the following requirements:
1. Support multiple sorting algorithms (quicksort, mergesort, heapsort)
2. Handle edge cases (empty list, single element, duplicates, None values)
3. Add comprehensive error handling with custom exceptions
4. Include detailed docstrings with examples
5. Use type hints for all parameters and return values
6. Add logging for debugging purposes
7. Optimize for both time and space complexity
8. Write unit tests covering all edge cases
9. Follow PEP 8 style guidelines
10. Add performance benchmarks

Example usage:
- sort([3, 1, 4, 1, 5], algorithm='quicksort')
- sort([], algorithm='mergesort') should return []
- sort([None, 1, 2]) should raise ValueError

Additional considerations:
- Thread safety for concurrent usage
- Memory efficiency for large datasets
- Compatibility with Python 3.8+
- Integration with common data science libraries

Please provide production-ready, fully documented code.
`.trim()
        }
    ];
    
    console.log('\n📊 تحليل Regularization:');
    console.log('─'.repeat(70));
    
    prompts.forEach(({ name, text }) => {
        const penalty = calculateRegularization(text);
        const length = text.length;
        const tokens = Math.ceil(length / 4);
        
        console.log(`\n${name}:`);
        console.log(`  الطول:               ${length} أحرف`);
        console.log(`  التوكنات (تقديري):  ${tokens} tokens`);
        console.log(`  Regularization:      ${penalty.toFixed(4)}`);
        
        if (penalty > 0.5) {
            console.log(`  التقييم:             ❌ معقد جداً - يحتاج تبسيط`);
        } else if (penalty > 0.2) {
            console.log(`  التقييم:             ⚠️ معقد - يمكن تحسينه`);
        } else {
            console.log(`  التقييم:             ✅ بسيط وواضح`);
        }
    });
    
    // تطبيق التبسيط على البرومبت المعقد
    const complexPrompt = prompts[2].text;
    console.log('\n\n📝 تطبيق التبسيط على البرومبت المعقد:');
    console.log('─'.repeat(70));
    
    const reductions = [0.2, 0.4, 0.6];
    reductions.forEach(reduction => {
        const simplified = simplifyPrompt(complexPrompt, reduction);
        const originalTokens = Math.ceil(complexPrompt.length / 4);
        const simplifiedTokens = Math.ceil(simplified.length / 4);
        const actualReduction = ((originalTokens - simplifiedTokens) / originalTokens * 100).toFixed(1);
        
        console.log(`\nتقليل ${(reduction * 100).toFixed(0)}% (فعلي: ${actualReduction}%):`);
        console.log(`  الطول الأصلي:  ${complexPrompt.length} أحرف (${originalTokens} tokens)`);
        console.log(`  الطول الجديد:  ${simplified.length} أحرف (${simplifiedTokens} tokens)`);
        console.log(`  النسبة:         ${((simplified.length / complexPrompt.length) * 100).toFixed(1)}%`);
    });
    
    // عرض البرومبت المبسط
    console.log('\n\n📄 البرومبت المبسط (تقليل 40%):');
    console.log('─'.repeat(70));
    const simplified = simplifyPrompt(complexPrompt, 0.4);
    console.log(simplified);
    console.log('─'.repeat(70));
}

// ==================== DEMO 6: Comprehensive Analysis ====================

export async function demo6_comprehensiveAnalysis() {
    console.log('\n' + '='.repeat(70));
    console.log('🔬 DEMO 6: Comprehensive Overfitting Analysis');
    console.log('='.repeat(70));
    
    const testCases = createSampleTestCases();
    const executor = createMockExecutor(0.6); // High overfitting
    
    const prompt = `
You are an expert AI assistant specialized in software development.
Follow these guidelines strictly:
- Write clean, maintainable code
- Add comprehensive error handling
- Include detailed comments
- Follow best practices for the language
- Consider edge cases
- Optimize performance
- Use design patterns appropriately
- Write unit tests
- Document all functions
- Follow style guidelines
`;
    
    console.log('\n🔍 بدء التحليل الشامل...');
    console.log('هذا قد يستغرق بضع دقائق...\n');
    
    const analysis = await comprehensiveOverfittingAnalysis(
        prompt,
        testCases,
        executor
    );
    
    // عرض النتائج
    console.log('\n📊 النتائج الشاملة:');
    console.log('═'.repeat(70));
    
    // 1. Overfitting Report
    console.log('\n1️⃣ تقرير Overfitting:');
    printOverfittingReport(analysis.overfittingReport);
    
    // 2. Cross Validation
    console.log('\n2️⃣ Cross Validation:');
    console.log(`   متوسط النقاط:  ${(analysis.crossValidation.meanScore * 100).toFixed(1)}%`);
    console.log(`   الاستقرار:     ${analysis.crossValidation.isStable ? '✅' : '❌'}`);
    console.log(`   Std Dev:       ${(analysis.crossValidation.stdDeviation * 100).toFixed(1)}%`);
    
    // 3. Held-out Validation
    console.log('\n3️⃣ Held-out Validation:');
    console.log(`   Train:         ${(analysis.heldOutValidation.trainScore * 100).toFixed(1)}%`);
    console.log(`   Val:           ${(analysis.heldOutValidation.valScore * 100).toFixed(1)}%`);
    console.log(`   Test:          ${(analysis.heldOutValidation.testScore * 100).toFixed(1)}%`);
    console.log(`   Gap:           ${(analysis.heldOutValidation.trainTestGap * 100).toFixed(1)}%`);
    console.log(`   Generalization: ${(analysis.heldOutValidation.generalizationScore * 100).toFixed(1)}%`);
    
    // 4. Diversity
    console.log('\n4️⃣ تحليل التنوع:');
    console.log(`   التنوع:        ${(analysis.diversityAnalysis.diversityScore * 100).toFixed(1)}%`);
    console.log(`   الفئات:        ${analysis.diversityAnalysis.uniqueCategories}`);
    console.log(`   كافٍ:          ${analysis.diversityAnalysis.isSufficientlyDiverse ? '✅' : '❌'}`);
    
    // 5. Regularization
    console.log('\n5️⃣ Regularization:');
    console.log(`   Penalty:       ${analysis.regularizationPenalty.toFixed(4)}`);
    
    // 6. Simplified Prompt
    if (analysis.simplifiedPrompt) {
        console.log('\n6️⃣ البرومبت المبسط:');
        console.log('─'.repeat(70));
        console.log(analysis.simplifiedPrompt);
        console.log('─'.repeat(70));
        
        const originalTokens = Math.ceil(prompt.length / 4);
        const simplifiedTokens = Math.ceil(analysis.simplifiedPrompt.length / 4);
        const reduction = ((originalTokens - simplifiedTokens) / originalTokens * 100).toFixed(1);
        console.log(`\n   التقليل: ${reduction}% (${originalTokens} → ${simplifiedTokens} tokens)`);
    }
    
    console.log('\n═'.repeat(70));
    console.log('✅ التحليل الشامل اكتمل');
    console.log('═'.repeat(70));
}

// ==================== Main Runner ====================

export async function runAllDemos() {
    console.log('\n' + '█'.repeat(70));
    console.log('🎯 DIRECTIVE-038: Prompt Overfitting Detection System');
    console.log('█'.repeat(70));
    console.log('\nنظام متكامل للكشف عن ومعالجة Overfitting في البرومبتات المُحسّنة');
    
    try {
        await demo1_basicDetection();
        await demo2_crossValidation();
        await demo3_heldOutValidation();
        await demo4_diversityAnalysis();
        await demo5_regularization();
        await demo6_comprehensiveAnalysis();
        
        console.log('\n' + '█'.repeat(70));
        console.log('✅ جميع العروض التوضيحية اكتملت بنجاح!');
        console.log('█'.repeat(70));
        
    } catch (error) {
        console.error('\n❌ خطأ في تشغيل العروض:', error);
    }
}

// تشغيل تلقائي إذا تم استدعاء الملف مباشرة
if (require.main === module) {
    runAllDemos().catch(console.error);
}
