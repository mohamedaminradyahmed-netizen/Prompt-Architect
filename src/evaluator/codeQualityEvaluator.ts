/**
 * Code Quality Evaluator (DIRECTIVE-016)
 *
 * لماذا؟
 * نحتاج مقاييس “قابلة للحوسبة” لتقييم جودة الكود المُولّد بدون الاعتماد على الانطباعات.
 * المشروع لا يحتوي ESLint/TSLint حالياً، لذا نستخدم TypeScript compiler + قواعد تحليل ثابت
 * بسيطة كبديل خفيف (بدون إضافة تبعيات جديدة).
 */

import ts from 'typescript';

export interface SecurityIssue {
  type: string;
  severity: 'low' | 'medium' | 'high';
  description: string;
  location: string;
}

export interface CodeQualityMetrics {
  syntaxScore: number; // 0-100
  bestPracticesScore: number;
  hasTests: boolean;
  documentationScore: number;
  securityIssues: SecurityIssue[];
  performanceScore: number;
  overallScore: number;
}

function clamp(n: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, n));
}

function firstMatchLocation(text: string, re: RegExp): string {
  const m = text.match(re);
  if (!m || m.index == null) return 'n/a';
  return `index:${m.index}`;
}

function evaluateSyntax(code: string, language: string): { score: number; errors: string[] } {
  const lang = language.toLowerCase();

  if (lang.includes('ts')) {
    const result = ts.transpileModule(code, {
      compilerOptions: {
        target: ts.ScriptTarget.ES2020,
        module: ts.ModuleKind.ESNext,
        strict: true,
      },
      reportDiagnostics: true,
    });

    const diagnostics = result.diagnostics || [];
    const errors = diagnostics
      .filter((d) => d.category === ts.DiagnosticCategory.Error)
      .map((d) => ts.flattenDiagnosticMessageText(d.messageText, '\n'));

    // كل خطأ يخفض الثقة بشكل واضح، لكن لا نُصفّر فوراً.
    const penalty = errors.length === 0 ? 0 : Math.min(90, 25 + errors.length * 15);
    return { score: clamp(100 - penalty, 0, 100), errors };
  }

  // JavaScript (تقريب): محاولة “compile” عبر Function.
  try {
    // eslint-disable-next-line no-new-func
    new Function(code);
    return { score: 100, errors: [] };
  } catch (e: any) {
    return { score: 10, errors: [String(e?.message || e)] };
  }
}

function evaluateBestPractices(code: string): number {
  let score = 80;

  const penalties: Array<{ re: RegExp; points: number; why: string }> = [
    { re: /\bvar\b/g, points: 10, why: 'استخدم let/const بدلاً من var.' },
    { re: /[^=!]==[^=]/g, points: 10, why: 'استخدم === بدلاً من ==.' },
    { re: /\bany\b/g, points: 8, why: 'تجنب any قدر الإمكان.' },
    { re: /\bconsole\.log\b/g, points: 5, why: 'تجنب console.log في كود الإنتاج.' },
    { re: /\bTODO\b/g, points: 3, why: 'لا تترك TODO داخل الشيفرة النهائية.' },
  ];

  for (const p of penalties) {
    if (p.re.test(code)) score -= p.points;
  }

  // مكافآت بسيطة
  if (/\btry\s*\{[\s\S]*?\}\s*catch\s*\(/.test(code)) score += 5;
  if (/\btype\b|\binterface\b/.test(code)) score += 3;

  return clamp(score, 0, 100);
}

function evaluateHasTests(code: string): boolean {
  return /(\bdescribe\s*\(|\bit\s*\(|\btest\s*\()/.test(code);
}

function evaluateDocumentation(code: string): number {
  // JSDoc أو تعليقات كافية.
  const hasJsDoc = /\/\*\*[\s\S]*?\*\//.test(code);
  const lineCount = Math.max(1, code.split('\n').length);
  const commentLines = (code.match(/(^\s*\/\/.*$)/gm) || []).length;
  const ratio = commentLines / lineCount;

  let score = 40;
  if (hasJsDoc) score += 30;
  if (ratio >= 0.08) score += 30;
  else if (ratio >= 0.03) score += 15;

  return clamp(score, 0, 100);
}

function evaluateSecurity(code: string): SecurityIssue[] {
  const issues: SecurityIssue[] = [];

  const patterns: Array<{ re: RegExp; type: string; severity: SecurityIssue['severity']; desc: string }> = [
    { re: /\beval\s*\(/g, type: 'eval', severity: 'high', desc: 'استخدام eval يفتح باب حقن كود.' },
    { re: /dangerouslySetInnerHTML/g, type: 'xss', severity: 'high', desc: 'dangerouslySetInnerHTML قد يسبب XSS.' },
    { re: /\.innerHTML\s*=\s*/g, type: 'xss', severity: 'high', desc: 'تعيين innerHTML قد يسبب XSS.' },
    { re: /child_process\.(exec|execSync|spawn|spawnSync)\s*\(/g, type: 'rce', severity: 'high', desc: 'تنفيذ أوامر نظام قد يسبب RCE إذا كانت المدخلات غير موثوقة.' },
    { re: /\bpassword\b|\bapi[_-]?key\b|\bsecret\b/gi, type: 'secrets', severity: 'medium', desc: 'مؤشر محتمل لتعامل مع أسرار داخل الشيفرة.' },
  ];

  for (const p of patterns) {
    const m = code.match(p.re);
    if (m) {
      issues.push({
        type: p.type,
        severity: p.severity,
        description: p.desc,
        location: firstMatchLocation(code, p.re),
      });
    }
  }

  return issues;
}

function evaluatePerformance(code: string): number {
  let score = 70;

  // عقوبات heuristic: تداخل حلقات واضح
  const nestedLoops = /for\s*\([\s\S]*?\)\s*\{[\s\S]*?for\s*\(/.test(code) || /while\s*\([\s\S]*?\)\s*\{[\s\S]*?while\s*\(/.test(code);
  if (nestedLoops) score -= 20;

  // مكافآت بسيطة: استخدام Map/Set بدل بحث خطي متكرر
  if (/\bnew\s+(Map|Set)\b/.test(code)) score += 10;

  return clamp(score, 0, 100);
}

export async function evaluateCodeQuality(code: string, language: string): Promise<CodeQualityMetrics> {
  const syntax = evaluateSyntax(code, language);
  const bestPracticesScore = evaluateBestPractices(code);
  const hasTests = evaluateHasTests(code);
  const documentationScore = evaluateDocumentation(code);
  const securityIssues = evaluateSecurity(code);
  const performanceScore = evaluatePerformance(code);

  const securityPenalty = securityIssues.reduce((acc, i) => {
    if (i.severity === 'high') return acc + 25;
    if (i.severity === 'medium') return acc + 10;
    return acc + 3;
  }, 0);

  const overallScore = clamp(
    Math.round(
      0.30 * syntax.score +
        0.20 * bestPracticesScore +
        0.15 * documentationScore +
        0.15 * performanceScore +
        0.20 * (hasTests ? 100 : 40) -
        securityPenalty
    ),
    0,
    100
  );

  return {
    syntaxScore: syntax.score,
    bestPracticesScore,
    hasTests,
    documentationScore,
    securityIssues,
    performanceScore,
    overallScore,
  };
}
