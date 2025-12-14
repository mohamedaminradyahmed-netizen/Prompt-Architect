/**
 * Safety Checker (DIRECTIVE-031)
 *
 * لماذا؟
 * قبل اعتماد أي variation يجب منع مسارات المخاطر الشائعة (حقن البرومبت، طلب بيانات حساسة،
 * محتوى ضار، وتحيّز واضح). هذا يقلل مخاطـر الاستغلال حتى لو كان التقييم “جيداً”.
 */

export interface SafetyViolation {
  type: 'injection' | 'sensitive_data' | 'harmful' | 'bias';
  severity: 'low' | 'medium' | 'high';
  description: string;
  location: string;
}

export interface SafetyReport {
  isSafe: boolean;
  violations: SafetyViolation[];
  confidence: number;
  recommendations: string[];
}

type ModerationSignal = {
  flagged: boolean;
  reasons: string[];
};

const log = {
  warn: (msg: string, meta?: unknown) => console.warn(`[safety] ${msg}`, meta ?? ''),
};

function addViolation(
  violations: SafetyViolation[],
  v: Omit<SafetyViolation, 'location'> & { location?: string }
) {
  violations.push({ ...v, location: v.location ?? 'n/a' });
}

function firstMatchLocation(text: string, re: RegExp): string {
  const m = text.match(re);
  if (!m || m.index == null) return 'n/a';
  return `index:${m.index}`;
}

function customRules(text: string): SafetyViolation[] {
  const t = text;
  const violations: SafetyViolation[] = [];

  // 1) Prompt injection (أمثلة شائعة)
  const injectionPatterns: Array<{ re: RegExp; desc: string; severity: SafetyViolation['severity'] }> = [
    { re: /\b(ignore|disregard)\b.*\b(previous|earlier)\b/i, desc: 'محاولة تعطيل تعليمات سابقة (prompt injection).', severity: 'high' },
    { re: /\b(system prompt|developer message|hidden instructions)\b/i, desc: 'محاولة الوصول لتعليمات النظام/المطور.', severity: 'high' },
    { re: /\b(do anything now|jailbreak)\b/i, desc: 'إشارة jailbreak صريحة.', severity: 'high' },
    { re: /\bact as\b.*\b(system|developer)\b/i, desc: 'محاولة انتحال دور النظام/المطور.', severity: 'medium' },
  ];
  for (const p of injectionPatterns) {
    if (p.re.test(t)) {
      addViolation(violations, {
        type: 'injection',
        severity: p.severity,
        description: p.desc,
        location: firstMatchLocation(t, p.re),
      });
    }
  }

  // 2) Sensitive data requests
  const sensitivePatterns: Array<{ re: RegExp; desc: string }> = [
    { re: /\b(password|passcode|api\s*key|secret\s*key|private\s*key|token)\b/i, desc: 'طلب/ذكر محتمل لبيانات سرية (كلمات مرور/مفاتيح/توكنات).' },
    { re: /\b(ssn|social\s*security|credit\s*card|cvv|bank\s*account)\b/i, desc: 'طلب/ذكر محتمل لبيانات مالية أو هوية.' },
    { re: /\b(address|phone\s*number|email)\b/i, desc: 'طلب/ذكر محتمل لبيانات شخصية (PII).' },
  ];
  for (const p of sensitivePatterns) {
    if (p.re.test(t)) {
      addViolation(violations, {
        type: 'sensitive_data',
        severity: 'high',
        description: p.desc,
        location: firstMatchLocation(t, p.re),
      });
    }
  }

  // 3) Harmful content (مؤشرات عامة)
  const harmfulPatterns: Array<{ re: RegExp; desc: string; severity: SafetyViolation['severity'] }> = [
    { re: /\b(how to|instructions|guide)\b.*\b(bomb|weapon|explosive)\b/i, desc: 'طلب إرشادات لتصنيع/استخدام سلاح.', severity: 'high' },
    { re: /\b(self-harm|suicide)\b/i, desc: 'مؤشر محتوى إيذاء النفس.', severity: 'high' },
    { re: /\b(hate|kill all)\b/i, desc: 'مؤشر خطاب كراهية/عنف.', severity: 'high' },
  ];
  for (const p of harmfulPatterns) {
    if (p.re.test(t)) {
      addViolation(violations, {
        type: 'harmful',
        severity: p.severity,
        description: p.desc,
        location: firstMatchLocation(t, p.re),
      });
    }
  }

  // 4) Bias detection (بدون نموذج، قواعد بسيطة)
  const biasPatterns: Array<{ re: RegExp; desc: string }> = [
    { re: /\b(all\s+(men|women|muslims|jews|christians|black\s+people|white\s+people))\b/i, desc: 'تعميم على مجموعة بشرية (تحيز محتمل).' },
    { re: /\b(inferior|superior)\b.*\b(race|religion|gender)\b/i, desc: 'لغة تفاضل/تحقير مرتبطة بهوية.' },
  ];
  for (const p of biasPatterns) {
    if (p.re.test(t)) {
      addViolation(violations, {
        type: 'bias',
        severity: 'medium',
        description: p.desc,
        location: firstMatchLocation(t, p.re),
      });
    }
  }

  return violations;
}

async function openAIModeration(text: string): Promise<ModerationSignal | null> {
  // DIRECTIVE-031 يطلب استخدام OpenAI Moderation API.
  // في بيئة بدون مفتاح، نرجع null ونعتمد على القواعد المحلية.
  const apiKey = (typeof process !== 'undefined' && process.env && process.env.OPENAI_API_KEY) ? process.env.OPENAI_API_KEY : undefined;
  if (!apiKey) return null;

  try {
    const res = await fetch('https://api.openai.com/v1/moderations', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({ model: 'omni-moderation-latest', input: text }),
    });

    if (!res.ok) {
      log.warn('Moderation API returned non-OK', { status: res.status });
      return null;
    }

    const data: any = await res.json();
    const r = data?.results?.[0];
    const flagged = Boolean(r?.flagged);
    const categories = r?.categories || {};
    const reasons = Object.entries(categories)
      .filter(([, v]) => Boolean(v))
      .map(([k]) => String(k));

    return { flagged, reasons };
  } catch (e) {
    log.warn('Moderation API call failed', e);
    return null;
  }
}

/**
 * فحص الأمان قبل قبول أي variation.
 */
export async function checkSafety(variation: string): Promise<SafetyReport> {
  const violations = customRules(variation);

  const moderation = await openAIModeration(variation);
  if (moderation?.flagged) {
    addViolation(violations, {
      type: 'harmful',
      severity: 'high',
      description: `OpenAI moderation flagged content: ${moderation.reasons.join(', ') || 'unknown'}`,
      location: 'moderation',
    });
  }

  const isSafe = violations.every((v) => v.severity !== 'high');

  const recommendations: string[] = [];
  if (!isSafe) {
    recommendations.push('ارفض variation أو اطلب إعادة صياغة مع إزالة المحتوى/الطلب عالي الخطورة.');
  }
  if (violations.some((v) => v.type === 'injection')) {
    recommendations.push('أعد كتابة البرومبت لمنع أي تعليمات تتجاوز السياسات أو تطلب تعليمات مخفية.');
  }
  if (violations.some((v) => v.type === 'sensitive_data')) {
    recommendations.push('امنع طلب/تخزين/عرض أي بيانات سرية أو PII، واستخدم masking في السجلات.');
  }

  // ثقة تقريبية: أعلى عندما تتفق القواعد + moderation (إن توفرت).
  const confidence = moderation ? 0.85 : 0.6;

  return { isSafe, violations, confidence, recommendations };
}
