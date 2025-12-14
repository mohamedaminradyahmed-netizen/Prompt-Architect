/**
 * Content Quality Evaluator
 * Specialized metrics for marketing and content evaluation
 */

export interface BrandVoice {
  tone: 'professional' | 'casual' | 'friendly' | 'authoritative' | 'playful';
  formality: 'formal' | 'informal' | 'mixed';
  personality: string[];
}

export interface ContentQualityMetrics {
  toneScore: number;
  readabilityScore: number;     // Flesch Reading Ease
  gradeLevel: number;           // Flesch-Kincaid Grade
  seoScore: number;
  hasCTA: boolean;
  ctaEffectiveness: number;
  emotionalScore: number;
  overallScore: number;
}

/**
 * Evaluate content quality for marketing/content writing
 */
export function evaluateContentQuality(
  content: string,
  targetAudience?: string,
  brandVoice?: BrandVoice
): ContentQualityMetrics {
  const toneScore = calculateToneConsistency(content, brandVoice);
  const readabilityScore = calculateFleschReadingEase(content);
  const gradeLevel = calculateFleschKincaidGrade(content);
  const seoScore = calculateSEOScore(content);
  const hasCTA = detectCallToAction(content);
  const ctaEffectiveness = hasCTA ? calculateCTAEffectiveness(content) : 0;
  const emotionalScore = calculateEmotionalAppeal(content);

  const overallScore = Math.round(
    (toneScore * 0.2) +
    (readabilityScore * 0.15) +
    (seoScore * 0.2) +
    (ctaEffectiveness * 0.2) +
    (emotionalScore * 0.25)
  );

  return {
    toneScore,
    readabilityScore,
    gradeLevel,
    seoScore,
    hasCTA,
    ctaEffectiveness,
    emotionalScore,
    overallScore,
  };
}

/**
 * Calculate tone consistency with brand voice
 */
function calculateToneConsistency(content: string, brandVoice?: BrandVoice): number {
  if (!brandVoice) return 75; // Default neutral score

  const words = content.toLowerCase().split(/\s+/);
  let score = 50;

  // Professional tone indicators
  const professionalWords = ['expertise', 'solution', 'professional', 'quality', 'reliable'];
  const casualWords = ['awesome', 'cool', 'hey', 'super', 'amazing'];
  const friendlyWords = ['welcome', 'help', 'support', 'together', 'community'];

  const professionalCount = words.filter(w => professionalWords.includes(w)).length;
  const casualCount = words.filter(w => casualWords.includes(w)).length;
  const friendlyCount = words.filter(w => friendlyWords.includes(w)).length;

  switch (brandVoice.tone) {
    case 'professional':
      score += professionalCount * 10 - casualCount * 5;
      break;
    case 'casual':
      score += casualCount * 10 - professionalCount * 3;
      break;
    case 'friendly':
      score += friendlyCount * 8 + casualCount * 5;
      break;
  }

  return Math.max(0, Math.min(100, score));
}

/**
 * Calculate Flesch Reading Ease score
 */
function calculateFleschReadingEase(content: string): number {
  const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0);
  const words = content.split(/\s+/).filter(w => w.length > 0);
  const syllables = words.reduce((count, word) => count + countSyllables(word), 0);

  if (sentences.length === 0 || words.length === 0) return 0;

  const avgSentenceLength = words.length / sentences.length;
  const avgSyllablesPerWord = syllables / words.length;

  const score = 206.835 - (1.015 * avgSentenceLength) - (84.6 * avgSyllablesPerWord);
  return Math.max(0, Math.min(100, Math.round(score)));
}

/**
 * Calculate Flesch-Kincaid Grade Level
 */
function calculateFleschKincaidGrade(content: string): number {
  const sentences = content.split(/[.!?]+/).filter(s => s.trim().length > 0);
  const words = content.split(/\s+/).filter(w => w.length > 0);
  const syllables = words.reduce((count, word) => count + countSyllables(word), 0);

  if (sentences.length === 0 || words.length === 0) return 0;

  const avgSentenceLength = words.length / sentences.length;
  const avgSyllablesPerWord = syllables / words.length;

  const grade = (0.39 * avgSentenceLength) + (11.8 * avgSyllablesPerWord) - 15.59;
  return Math.max(0, Math.round(grade * 10) / 10);
}

/**
 * Calculate basic SEO score
 */
function calculateSEOScore(content: string): number {
  let score = 0;
  const words = content.toLowerCase().split(/\s+/);
  const wordCount = words.length;

  // Length check (150-300 words is good for most content)
  if (wordCount >= 150 && wordCount <= 300) {
    score += 30;
  } else if (wordCount >= 100) {
    score += 20;
  }

  // Keyword density (simple check for repeated important words)
  const wordFreq = new Map<string, number>();
  words.forEach(word => {
    if (word.length > 3) {
      wordFreq.set(word, (wordFreq.get(word) || 0) + 1);
    }
  });

  const maxFreq = Math.max(...wordFreq.values());
  const keywordDensity = maxFreq / wordCount;
  
  if (keywordDensity >= 0.02 && keywordDensity <= 0.05) {
    score += 25; // Good keyword density
  }

  // Structure indicators
  if (content.includes('\n') || content.includes('•') || content.includes('-')) {
    score += 20; // Has structure
  }

  // Action words
  const actionWords = ['discover', 'learn', 'get', 'find', 'explore', 'start'];
  const hasActionWords = actionWords.some(word => content.toLowerCase().includes(word));
  if (hasActionWords) {
    score += 15;
  }

  // Questions (engagement)
  if (content.includes('?')) {
    score += 10;
  }

  return Math.min(100, score);
}

/**
 * Detect call-to-action
 */
function detectCallToAction(content: string): boolean {
  const ctaPatterns = [
    /\b(click|tap|visit|call|contact|subscribe|sign up|get started|learn more|buy now|order now|download|try)\b/i,
    /\b(join|register|apply|book|schedule|request|claim|grab|shop)\b/i,
  ];

  return ctaPatterns.some(pattern => pattern.test(content));
}

/**
 * Calculate CTA effectiveness
 */
function calculateCTAEffectiveness(content: string): number {
  let score = 0;

  // Strong action verbs
  const strongVerbs = ['get', 'start', 'discover', 'unlock', 'transform', 'boost'];
  const hasStrongVerbs = strongVerbs.some(verb => content.toLowerCase().includes(verb));
  if (hasStrongVerbs) score += 30;

  // Urgency words
  const urgencyWords = ['now', 'today', 'limited', 'exclusive', 'instant'];
  const hasUrgency = urgencyWords.some(word => content.toLowerCase().includes(word));
  if (hasUrgency) score += 25;

  // Benefit-focused
  const benefitWords = ['free', 'save', 'improve', 'increase', 'reduce', 'better'];
  const hasBenefits = benefitWords.some(word => content.toLowerCase().includes(word));
  if (hasBenefits) score += 25;

  // Clear and specific
  if (content.length < 200) score += 10; // Concise
  if (!/\b(maybe|might|could|perhaps)\b/i.test(content)) score += 10; // Decisive

  return Math.min(100, score);
}

/**
 * Calculate emotional appeal
 */
function calculateEmotionalAppeal(content: string): number {
  let score = 0;

  // Positive emotions
  const positiveWords = ['amazing', 'incredible', 'fantastic', 'wonderful', 'excellent', 'outstanding', 'perfect'];
  const positiveCount = positiveWords.filter(word => content.toLowerCase().includes(word)).length;
  score += Math.min(30, positiveCount * 10);

  // Power words
  const powerWords = ['proven', 'guaranteed', 'exclusive', 'secret', 'ultimate', 'revolutionary'];
  const powerCount = powerWords.filter(word => content.toLowerCase().includes(word)).length;
  score += Math.min(25, powerCount * 8);

  // Personal pronouns (connection)
  const personalPronouns = content.match(/\b(you|your|we|our|us)\b/gi) || [];
  score += Math.min(20, personalPronouns.length * 2);

  // Sensory words
  const sensoryWords = ['see', 'hear', 'feel', 'taste', 'touch', 'imagine', 'picture'];
  const sensoryCount = sensoryWords.filter(word => content.toLowerCase().includes(word)).length;
  score += Math.min(15, sensoryCount * 5);

  // Questions (engagement)
  const questionCount = (content.match(/\?/g) || []).length;
  score += Math.min(10, questionCount * 5);

  return Math.min(100, score);
}

/**
 * Count syllables in a word (approximation)
 */
function countSyllables(word: string): number {
  word = word.toLowerCase();
  if (word.length <= 3) return 1;
  
  const vowels = 'aeiouy';
  let count = 0;
  let previousWasVowel = false;

  for (let i = 0; i < word.length; i++) {
    const isVowel = vowels.includes(word[i]);
    if (isVowel && !previousWasVowel) {
      count++;
    }
    previousWasVowel = isVowel;
  }

  // Handle silent 'e'
  if (word.endsWith('e')) {
    count--;
  }

  return Math.max(1, count);
}