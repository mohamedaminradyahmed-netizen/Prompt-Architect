/**
 * Prisma Database Client (DIRECTIVE-046)
 *
 * هذا الملف يوفر نسخة واحدة (singleton) من Prisma Client
 * لضمان عدم إنشاء اتصالات متعددة بقاعدة البيانات.
 *
 * الاستخدام:
 * import { prisma } from './db/client';
 * const prompts = await prisma.prompt.findMany();
 *
 * ملاحظة: في Prisma 7، نستخدم @prisma/adapter-pg للاتصال بـ PostgreSQL
 */

import { Pool } from 'pg';
import { PrismaPg } from '@prisma/adapter-pg';
import { PrismaClient } from '../generated/prisma';

// تخزين النسخة في global لتجنب إنشاء اتصالات متعددة أثناء التطوير مع Hot Reload
const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
  pool: Pool | undefined;
};

/**
 * إنشاء Pool للاتصال بـ PostgreSQL
 */
function createPool(): Pool {
  const connectionString = process.env.DATABASE_URL;

  if (!connectionString) {
    throw new Error('DATABASE_URL environment variable is not set');
  }

  return new Pool({
    connectionString,
    max: 10, // الحد الأقصى لعدد الاتصالات
    idleTimeoutMillis: 30000, // مهلة الخمول بالميلي ثانية
    connectionTimeoutMillis: 2000, // مهلة الاتصال
  });
}

/**
 * إنشاء Prisma Client مع PostgreSQL adapter
 */
function createPrismaClient(): PrismaClient {
  const pool = globalForPrisma.pool ?? createPool();

  if (process.env.NODE_ENV !== 'production') {
    globalForPrisma.pool = pool;
  }

  const adapter = new PrismaPg(pool);

  return new PrismaClient({
    adapter,
    log:
      process.env.NODE_ENV === 'development'
        ? ['query', 'error', 'warn']
        : ['error'],
  });
}

/**
 * نسخة Prisma Client الرئيسية
 * تستخدم في جميع أنحاء التطبيق للتفاعل مع قاعدة البيانات
 */
export const prisma = globalForPrisma.prisma ?? createPrismaClient();

// في بيئة التطوير، نحفظ النسخة في global
if (process.env.NODE_ENV !== 'production') {
  globalForPrisma.prisma = prisma;
}

/**
 * دالة مساعدة للتحقق من صحة الاتصال بقاعدة البيانات
 */
export async function checkDatabaseConnection(): Promise<boolean> {
  try {
    await prisma.$queryRaw`SELECT 1`;
    return true;
  } catch (error) {
    console.error('Database connection failed:', error);
    return false;
  }
}

/**
 * دالة لإغلاق اتصال قاعدة البيانات
 * تُستخدم عند إيقاف التطبيق
 */
export async function disconnectDatabase(): Promise<void> {
  await prisma.$disconnect();
  if (globalForPrisma.pool) {
    await globalForPrisma.pool.end();
  }
}

/**
 * دالة للحصول على إحصائيات Pool
 */
export function getPoolStats(): {
  totalCount: number;
  idleCount: number;
  waitingCount: number;
} | null {
  if (!globalForPrisma.pool) {
    return null;
  }

  return {
    totalCount: globalForPrisma.pool.totalCount,
    idleCount: globalForPrisma.pool.idleCount,
    waitingCount: globalForPrisma.pool.waitingCount,
  };
}

// تصدير أنواع النماذج للاستخدام في أجزاء أخرى من التطبيق
export type {
  Prompt,
  Variation,
  Feedback,
  Lineage,
  TestCase,
  BenchmarkResult,
  AnalyticsEvent,
} from '../generated/prisma';

export default prisma;
