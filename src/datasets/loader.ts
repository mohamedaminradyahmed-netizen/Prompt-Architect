/**
 * Dataset loader for reference datasets used in prompt evaluation and testing
 * This module provides functions to load and access the reference datasets
 * for code generation, content writing, and marketing copy tasks.
 */

import codeData from './code.json';
import contentData from './content.json';
import marketingData from './marketing.json';

// Define interfaces for our dataset structures
export interface EvaluationCriteria {
  [key: string]: string;
}

export interface DatasetItem {
  id: string;
  prompt: string;
  expectedOutput: string;
  category: string;
  evaluationCriteria: EvaluationCriteria;
}

export interface CodeDatasetItem extends DatasetItem {
  category: 'CODE_GENERATION';
  evaluationCriteria: {
    correctness: string;
    syntax?: string;
    efficiency?: string;
    readability?: string;
    typescript?: string;
    regex?: string;
    edgeCases?: string;
    pythonic?: string;
    noBuiltin?: string;
    errorHandling?: string;
    twoPointer?: string;
    orderPreservation?: string;
    memory?: string;
    noRecursion?: string;
    precision?: string;
    length?: string;
    recursion?: string;
    objectHandling?: string;
    duplicates?: string;
    order?: string;
    encapsulation?: string;
    initialValue?: string;
  };
}

export interface ContentDatasetItem extends DatasetItem {
  category: 'CONTENT_WRITING';
  evaluationCriteria: {
    tone: string;
    structure?: string;
    readability?: string;
    engagement?: string;
    features?: string;
    benefits?: string;
    technical?: string;
    simplicity?: string;
    analogies?: string;
    applications?: string;
    hashtags?: string;
    actionable?: string;
    details?: string;
    balance?: string;
    helpful?: string;
    community?: string;
    inclusivity?: string;
    practical?: string;
    objectivity?: string;
    comparisons?: string;
    recommendations?: string;
    emotional?: string;
    ongoing?: string;
    aspirational?: string;
    beautiful?: string;
  };
}

export interface MarketingDatasetItem extends DatasetItem {
  category: 'MARKETING_COPY';
  evaluationCriteria: {
    local?: string;
    specials?: string;
    social?: string;
    atmosphere?: string;
    convenience?: string;
    services?: string;
    guarantee?: string;
    special?: string;
    values?: string;
    transparency?: string;
    impact?: string;
    community?: string;
    credibility?: string;
    results?: string;
    specialization?: string;
    accessibility?: string;
    thread?: string;
    engagement?: string;
    clarity?: string;
    conversion?: string;
    culture?: string;
    experience?: string;
    painPoints?: string;
    process?: string;
    access?: string;
    specificity?: string;
    enthusiasm?: string;
    helpful?: string;
    inclusive?: string;
  };
}

export type DatasetType = CodeDatasetItem | ContentDatasetItem | MarketingDatasetItem;

export interface Datasets {
  code: CodeDatasetItem[];
  content: ContentDatasetItem[];
  marketing: MarketingDatasetItem[];
}

// Load the datasets
const datasets: Datasets = {
  code: codeData as CodeDatasetItem[],
  content: contentData as ContentDatasetItem[],
  marketing: marketingData as MarketingDatasetItem[]
};

/**
 * Get all datasets
 */
export function getAllDatasets(): Datasets {
  return datasets;
}

/**
 * Get a specific dataset by type
 */
export function getDataset(type: keyof Datasets): DatasetType[] {
  return datasets[type];
}

/**
 * Get a specific item by ID from any dataset
 */
export function getItemById(id: string): DatasetType | undefined {
  // Check code dataset
  const codeItem = datasets.code.find(item => item.id === id);
  if (codeItem) return codeItem;
  
  // Check content dataset
  const contentItem = datasets.content.find(item => item.id === id);
  if (contentItem) return contentItem;
  
  // Check marketing dataset
  const marketingItem = datasets.marketing.find(item => item.id === id);
  if (marketingItem) return marketingItem;
  
  return undefined;
}

/**
 * Get items by category
 */
export function getItemsByCategory(category: string): DatasetType[] {
  const allItems = [
    ...datasets.code,
    ...datasets.content,
    ...datasets.marketing
  ];
  
  return allItems.filter(item => item.category === category);
}

/**
 * Get random items from a dataset
 */
export function getRandomItems(type: keyof Datasets, count: number): DatasetType[] {
  const dataset = datasets[type];
  const shuffled = [...dataset].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, count);
}

/**
 * Get evaluation criteria for a specific item
 */
export function getEvaluationCriteria(id: string): EvaluationCriteria | undefined {
  const item = getItemById(id);
  return item?.evaluationCriteria;
}

/**
 * Validate if a prompt matches the expected category
 */
export function validateCategoryMatch(prompt: string, category: string): boolean {
  const items = getItemsByCategory(category);
  if (items.length === 0) return false;
  
  // Simple validation: check if prompt contains category keywords
  const categoryKeywords = {
    'CODE_GENERATION': ['function', 'code', 'programming', 'algorithm', 'implement'],
    'CONTENT_WRITING': ['write', 'article', 'blog', 'content', 'description'],
    'MARKETING_COPY': ['ad', 'marketing', 'promotional', 'social media', 'copy']
  };
  
  const keywords = categoryKeywords[category as keyof typeof categoryKeywords] || [];
  return keywords.some(keyword => prompt.toLowerCase().includes(keyword));
}

/**
 * Get dataset statistics
 */
export function getDatasetStats() {
  return {
    code: {
      total: datasets.code.length,
      categories: [...new Set(datasets.code.map(item => item.category))].length
    },
    content: {
      total: datasets.content.length,
      categories: [...new Set(datasets.content.map(item => item.category))].length
    },
    marketing: {
      total: datasets.marketing.length,
      categories: [...new Set(datasets.marketing.map(item => item.category))].length
    },
    overall: {
      total: datasets.code.length + datasets.content.length + datasets.marketing.length,
      categories: [...new Set([
        ...datasets.code,
        ...datasets.content,
        ...datasets.marketing
      ].map(item => item.category))].length
    }
  };
}

// Export the datasets for direct access
export { codeData, contentData, marketingData };
