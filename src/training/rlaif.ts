/**
 * DIRECTIVE-035: RLAIF (Reinforcement Learning from AI Feedback)
 * 
 * يقلل الاعتماد على البشر باستخدام AI للتقييم عبر:
 * 1. Bootstrap من بيانات بشرية أولية
 * 2. Self-Play Loop للتحسين المستمر
 * 3. Human-in-the-Loop Validation الدوري
 * 
 * لماذا: يسمح بالنطاق الكبير للتحسين مع الحفاظ على جودة عالية عبر المراجعة البشرية الدورية
 */

import { RewardModel, TrainingExample as RewardTrainingExample } from '../models/rewardModel';
import { PromptVariation, MutationType, mutationTypes, tryCatchStyleMutation, reduceContextMutation, expandMutation, constrainMutation } from '../mutations';
import { classifyPrompt, PromptCategory } from '../types/promptTypes';
import { getFeedbackFromStorage, HumanFeedback } from '../api/feedback';
import { collectTrainingData, TrainingExample } from './dataCollection';

// ============================================================================
// INTERFACES
// ============================================================================

/**
 * Policy interface: يحدد كيف نختار mutations
 */
export interface MutationPolicy {
  /**
   * يختار mutation type بناءً على prompt
   */
  selectMutation(prompt: string, availableMutations: MutationType[]): MutationType;
  
  /**
   * يحسّن Policy بناءً على experiences
   */
  update(rewards: Map<MutationType, number[]>): void;
  
  /**
   * يحصل على احتمالات الـ mutations (للتحليل)
   */
  getProbabilities(prompt: string, mutations: MutationType[]): Map<MutationType, number>;
}

/**
 * نتيجة تدريب RLAIF
 */
export interface ImprovedPolicy extends MutationPolicy {
  improvementStats: {
    startingAverageReward: number;
    endingAverageReward: number;
    iterations: number;
    humanValidations: number;
    humanCorrections: number;
  };
}

/**
 * إعدادات RLAIF Training
 */
export interface RLAIFConfig {
  iterations: number;              // عدد دورات Self-Play
  batchSize: number;               // عدد variations لكل iteration
  humanValidationInterval: number; // مراجعة بشرية كل N iterations
  bootstrapFromHumanFeedback: boolean; // استخدام human feedback للبداية
  minHumanFeedbackSamples: number; // الحد الأدنى من human feedback للبدء
}

// ============================================================================
// DEFAULT POLICY (Simple Probability-Based)
// ============================================================================

/**
 * Policy بسيط يعتمد على احتمالات متساوية
 */
export class DefaultMutationPolicy implements MutationPolicy {
  private mutationRewards: Map<MutationType, number[]>;
  
  constructor() {
    this.mutationRewards = new Map();
    mutationTypes.forEach(m => {
      this.mutationRewards.set(m, []);
    });
  }
  
  selectMutation(prompt: string, availableMutations: MutationType[]): MutationType {
    const probs = this.getProbabilities(prompt, availableMutations);
    
    // Sample from distribution
    const rand = Math.random();
    let cumulative = 0;
    for (const [mutation, prob] of probs.entries()) {
      cumulative += prob;
      if (rand < cumulative) {
        return mutation;
      }
    }
    
    return availableMutations[availableMutations.length - 1];
  }
  
  getProbabilities(prompt: string, mutations: MutationType[]): Map<MutationType, number> {
    const probs = new Map<MutationType, number>();
    const uniformProb = 1.0 / mutations.length;
    
    // إذا لم يكن لدينا بيانات، استخدم احتمالات متساوية
    let hasData = false;
    mutations.forEach(m => {
      const rewards = this.mutationRewards.get(m) || [];
      if (rewards.length > 0) hasData = true;
    });
    
    if (!hasData) {
      mutations.forEach(m => probs.set(m, uniformProb));
      return probs;
    }
    
    // احسب متوسط reward لكل mutation
    const avgRewards = new Map<MutationType, number>();
    mutations.forEach(m => {
      const rewards = this.mutationRewards.get(m) || [];
      if (rewards.length === 0) {
        avgRewards.set(m, 0.5); // Default neutral
      } else {
        const avg = rewards.reduce((a, b) => a + b, 0) / rewards.length;
        avgRewards.set(m, avg);
      }
    });
    
    // Convert to probabilities using softmax
    const exps = new Map<MutationType, number>();
    let sumExp = 0;
    mutations.forEach(m => {
      const exp = Math.exp(avgRewards.get(m)! * 5); // Temperature scaling
      exps.set(m, exp);
      sumExp += exp;
    });
    
    mutations.forEach(m => {
      probs.set(m, (exps.get(m)! / sumExp) || uniformProb);
    });
    
    return probs;
  }
  
  update(rewards: Map<MutationType, number[]>): void {
    // Merge new rewards
    rewards.forEach((newRewards, mutation) => {
      const existing = this.mutationRewards.get(mutation) || [];
      this.mutationRewards.set(mutation, [...existing, ...newRewards]);
      
      // Keep only last 100 rewards per mutation (sliding window)
      if (this.mutationRewards.get(mutation)!.length > 100) {
        const all = this.mutationRewards.get(mutation)!;
        this.mutationRewards.set(mutation, all.slice(-100));
      }
    });
  }
}

// ============================================================================
// BOOTSTRAP FROM HUMAN FEEDBACK
// ============================================================================

/**
 * Bootstrap Reward Model من human feedback
 */
async function bootstrapRewardModel(
  rewardModel: RewardModel,
  minSamples: number
): Promise<{ success: boolean; samplesUsed: number }> {
  // جمع بيانات التدريب من مصادر متعددة
  const trainingExamples: RewardTrainingExample[] = [];
  
  // 1. جمع من human feedback مباشرة
  const feedbacks = getFeedbackFromStorage();
  for (const feedback of feedbacks) {
    // نحتاج original prompt - قد نحتاج لتخزينه مع variation
    // للآن سنستخدم feedback كتقريب
    if (feedback.score >= 1 && feedback.score <= 5) {
      trainingExamples.push({
        id: feedback.id || `feedback_${Date.now()}`,
        originalPrompt: feedback.promptId, // TODO: يجب تخزين original prompt
        modifiedPrompt: feedback.variationId,
        outputs: { original: '', modified: '' },
        humanScore: feedback.score,
        metadata: {
          category: classifyPrompt(feedback.variationId).category,
          mutationType: 'unknown',
          timestamp: feedback.timestamp || new Date(),
          userId: feedback.userId,
        },
      });
    }
  }
  
  // 2. جمع من training data collection
  for await (const example of collectTrainingData()) {
    if (example.humanScore >= 1 && example.humanScore <= 5) {
      trainingExamples.push({
        id: example.id,
        originalPrompt: example.originalPrompt,
        modifiedPrompt: example.modifiedPrompt,
        context: example.context,
        outputs: example.outputs,
        humanScore: example.humanScore,
        feedback: example.feedback,
        metadata: example.metadata,
      });
    }
  }
  
  if (trainingExamples.length < minSamples) {
    return { success: false, samplesUsed: trainingExamples.length };
  }
  
  // تدريب Reward Model
  try {
    rewardModel.train(trainingExamples);
    return { success: true, samplesUsed: trainingExamples.length };
  } catch (error) {
    console.error('Failed to train reward model during bootstrap:', error);
    return { success: false, samplesUsed: trainingExamples.length };
  }
}

// ============================================================================
// APPLY MUTATION
// ============================================================================

/**
 * يطبق mutation على prompt ويعيد PromptVariation
 */
function applyMutation(prompt: string, mutationType: MutationType): PromptVariation {
  const category = classifyPrompt(prompt).category;
  
  switch (mutationType) {
    case 'try-catch-style':
      return tryCatchStyleMutation(prompt);
    case 'context-reduction':
      return reduceContextMutation(prompt);
    case 'expansion':
      return expandMutation(prompt);
    case 'constraint-addition':
      return constrainMutation(prompt, category);
    default:
      // Fallback: return original
      return {
        text: prompt,
        mutationType: 'try-catch-style',
        changeDescription: 'No mutation applied',
        expectedImpact: {},
      };
  }
}

// ============================================================================
// HUMAN-IN-THE-LOOP VALIDATION
// ============================================================================

/**
 * Human Validation: يعرض variations للمراجعة البشرية
 */
export interface HumanValidationResult {
  validated: boolean;
  corrections: Array<{ variation: PromptVariation; correctReward: number }>;
  avgHumanReward: number;
}

/**
 * يحاكي human validation (في الإنتاج، سيكون UI حقيقي)
 */
async function humanValidate(
  variations: Array<{ variation: PromptVariation; aiReward: number }>,
  rewardModel: RewardModel
): Promise<HumanValidationResult> {
  // في الإنتاج الحقيقي، سيتم عرض هذه في UI
  // للآن، نستخدم feedback موجود للتحقق
  
  const corrections: Array<{ variation: PromptVariation; correctReward: number }> = [];
  let totalReward = 0;
  let validatedCount = 0;
  
  for (const item of variations) {
    // ابحث عن human feedback لهذا variation
    const feedbacks = getFeedbackFromStorage();
    const relevantFeedback = feedbacks.find(
      f => f.variationId === item.variation.text.substring(0, 50)
    );
    
    if (relevantFeedback) {
      const humanReward = relevantFeedback.score / 5.0; // Normalize to 0-1
      const aiReward = item.aiReward;
      
      // إذا كان الفرق كبير، أضف correction
      if (Math.abs(humanReward - aiReward) > 0.2) {
        corrections.push({
          variation: item.variation,
          correctReward: humanReward,
        });
        
        // حدّث reward model بهذه التصحيحات
        const category = classifyPrompt(item.variation.text).category;
        // Note: نحتاج original prompt هنا - للبساطة سنستخدم variation
        rewardModel.train([{
          id: `correction_${Date.now()}`,
          originalPrompt: item.variation.text, // Simplified
          modifiedPrompt: item.variation.text,
          outputs: { original: '', modified: '' },
          humanScore: relevantFeedback.score,
          metadata: {
            category,
            mutationType: item.variation.mutationType,
            timestamp: new Date(),
            userId: relevantFeedback.userId,
          },
        }]);
      }
      
      totalReward += humanReward;
      validatedCount++;
    } else {
      // بدون feedback بشري، استخدم AI reward
      totalReward += item.aiReward;
      validatedCount++;
    }
  }
  
  return {
    validated: validatedCount > 0,
    corrections,
    avgHumanReward: validatedCount > 0 ? totalReward / validatedCount : 0,
  };
}

// ============================================================================
// MAIN RLAIF TRAINING FUNCTION
// ============================================================================

/**
 * RLAIF Training Loop
 * 
 * الاستراتيجية:
 * 1. Bootstrap من human feedback إذا كان متاحاً
 * 2. Self-Play Loop:
 *    - ولّد variations باستخدام Policy
 *    - قيّمها بـ Reward Model
 *    - حسّن Policy بناءً على النتائج
 * 3. Human Validation الدوري:
 *    - راجع عينات دورياً مع بشر
 *    - صحّح أخطاء Reward Model
 *    - أعد تدريب النموذج
 */
export async function rlaifTraining(
  initialPolicy: MutationPolicy,
  rewardModel: RewardModel,
  config: Partial<RLAIFConfig> = {}
): Promise<ImprovedPolicy> {
  const fullConfig: RLAIFConfig = {
    iterations: 10,
    batchSize: 20,
    humanValidationInterval: 3,
    bootstrapFromHumanFeedback: true,
    minHumanFeedbackSamples: 10,
    ...config,
  };
  
  console.log('🚀 Starting RLAIF Training...');
  console.log(`Config: ${JSON.stringify(fullConfig, null, 2)}`);
  
  // Wrap policy in ImprovedPolicy
  const improvedPolicy: ImprovedPolicy = {
    ...initialPolicy,
    improvementStats: {
      startingAverageReward: 0,
      endingAverageReward: 0,
      iterations: fullConfig.iterations,
      humanValidations: 0,
      humanCorrections: 0,
    },
  };
  
  // 1. BOOTSTRAP PHASE
  if (fullConfig.bootstrapFromHumanFeedback) {
    console.log('📚 Bootstrap: Training Reward Model from Human Feedback...');
    const bootstrapResult = await bootstrapRewardModel(rewardModel, fullConfig.minHumanFeedbackSamples);
    
    if (bootstrapResult.success) {
      console.log(`✅ Bootstrap successful: Used ${bootstrapResult.samplesUsed} human feedback samples`);
    } else {
      console.warn(`⚠️ Bootstrap incomplete: Only ${bootstrapResult.samplesUsed} samples (minimum: ${fullConfig.minHumanFeedbackSamples})`);
    }
  }
  
  // 2. SELF-PLAY LOOP
  let currentPrompt = "Write a function to process user input";
  const allRewards: number[] = [];
  
  for (let iteration = 0; iteration < fullConfig.iterations; iteration++) {
    console.log(`\n🔄 Iteration ${iteration + 1}/${fullConfig.iterations}`);
    
    const batchRewards = new Map<MutationType, number[]>();
    const batchVariations: Array<{ variation: PromptVariation; aiReward: number; mutationType: MutationType }> = [];
    
    // Generate batch of variations
    for (let b = 0; b < fullConfig.batchSize; b++) {
      // Select mutation using policy
      const selectedMutation = improvedPolicy.selectMutation(currentPrompt, mutationTypes);
      
      // Apply mutation
      const variation = applyMutation(currentPrompt, selectedMutation);
      
      // Evaluate with Reward Model
      const category = classifyPrompt(variation.text).category;
      const prediction = rewardModel.predict(currentPrompt, variation.text, variation.mutationType, category);
      const reward = prediction.score; // Normalize to 0-1
      
      // Store results
      const existingRewards = batchRewards.get(selectedMutation) || [];
      batchRewards.set(selectedMutation, [...existingRewards, reward]);
      batchVariations.push({ variation, aiReward: reward, mutationType: selectedMutation });
      allRewards.push(reward);
      
      // Update current prompt occasionally (self-play evolution)
      if (reward > 0.7) {
        currentPrompt = variation.text;
      }
    }
    
    // Calculate batch statistics
    const avgReward = allRewards.slice(-fullConfig.batchSize).reduce((a, b) => a + b, 0) / fullConfig.batchSize;
    if (iteration === 0) {
      improvedPolicy.improvementStats.startingAverageReward = avgReward;
    }
    
    console.log(`  Average Reward: ${avgReward.toFixed(4)}`);
    console.log(`  Mutations used: ${Array.from(batchRewards.keys()).join(', ')}`);
    
    // 3. HUMAN VALIDATION (Periodic)
    if ((iteration + 1) % fullConfig.humanValidationInterval === 0) {
      console.log(`  👤 Human Validation...`);
      
      // Select top variations for validation
      const topVariations = batchVariations
        .sort((a, b) => b.aiReward - a.aiReward)
        .slice(0, Math.min(5, batchVariations.length))
        .map(item => ({ variation: item.variation, aiReward: item.aiReward }));
      
      const validationResult = await humanValidate(topVariations, rewardModel);
      
      if (validationResult.validated) {
        improvedPolicy.improvementStats.humanValidations++;
        improvedPolicy.improvementStats.humanCorrections += validationResult.corrections.length;
        
        console.log(`    ✅ Validated: ${validationResult.avgHumanReward.toFixed(4)} avg reward`);
        console.log(`    📝 Corrections: ${validationResult.corrections.length}`);
      }
    }
    
    // Update policy based on rewards
    improvedPolicy.update(batchRewards);
  }
  
  // Final statistics
  const finalRewards = allRewards.slice(-fullConfig.batchSize);
  improvedPolicy.improvementStats.endingAverageReward = 
    finalRewards.reduce((a, b) => a + b, 0) / finalRewards.length;
  
  console.log('\n✅ RLAIF Training Complete!');
  console.log(`📊 Stats:`, improvedPolicy.improvementStats);
  
  return improvedPolicy;
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  rlaifTraining,
  DefaultMutationPolicy,
  bootstrapRewardModel,
  humanValidate,
};
