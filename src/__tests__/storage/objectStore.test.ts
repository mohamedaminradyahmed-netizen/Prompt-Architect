/**
 * Unit Tests for Object Storage System (DIRECTIVE-047)
 */

import { Readable } from 'stream';
import {
    createInMemoryStore,
    generateUniqueKey,
    parseStorageUri,
    getCategoryKey,
    StoragePaths,
} from '../../storage/objectStore';

describe('ObjectStore', () => {
    let store: ReturnType<typeof createInMemoryStore>;

    beforeEach(() => {
        store = createInMemoryStore('test-bucket');
    });

    afterEach(() => {
        store.clear();
    });

    describe('upload', () => {
        it('should upload a buffer and return URI', async () => {
            const data = Buffer.from('Hello, World!');
            const key = 'test/hello.txt';

            const uri = await store.upload(key, data);

            expect(uri).toBe('memory://test-bucket/test/hello.txt');
        });

        it('should upload with content type', async () => {
            const data = Buffer.from('{"key": "value"}');
            const key = 'test/data.json';

            const uri = await store.upload(key, data, {
                contentType: 'application/json',
            });

            expect(uri).toContain(key);

            const metadata = await store.getMetadata(key);
            expect(metadata.contentType).toBe('application/json');
        });

        it('should upload with custom metadata', async () => {
            const data = Buffer.from('test content');
            const key = 'test/metadata.txt';

            await store.upload(key, data, {
                metadata: { author: 'test-user', version: '1.0' },
            });

            const metadata = await store.getMetadata(key);
            expect(metadata.metadata).toEqual({ author: 'test-user', version: '1.0' });
        });

        it('should upload a stream', async () => {
            const content = 'Stream content here';
            const stream = Readable.from([content]);
            const key = 'test/stream.txt';

            await store.upload(key, stream);

            const downloaded = await store.download(key);
            expect(downloaded.toString()).toBe(content);
        });
    });

    describe('download', () => {
        it('should download an existing file', async () => {
            const content = 'Download test content';
            const key = 'test/download.txt';

            await store.upload(key, Buffer.from(content));
            const downloaded = await store.download(key);

            expect(downloaded.toString()).toBe(content);
        });

        it('should throw error for non-existent file', async () => {
            await expect(store.download('non-existent-key')).rejects.toThrow(
                'Object not found: non-existent-key'
            );
        });
    });

    describe('delete', () => {
        it('should delete an existing file', async () => {
            const key = 'test/to-delete.txt';
            await store.upload(key, Buffer.from('delete me'));

            expect(await store.exists(key)).toBe(true);

            await store.delete(key);

            expect(await store.exists(key)).toBe(false);
        });

        it('should not throw for non-existent file', async () => {
            await expect(store.delete('non-existent')).resolves.not.toThrow();
        });
    });

    describe('deleteMany', () => {
        it('should delete multiple files', async () => {
            const keys = ['test/file1.txt', 'test/file2.txt', 'test/file3.txt'];

            for (const key of keys) {
                await store.upload(key, Buffer.from(`content for ${key}`));
            }

            for (const key of keys) {
                expect(await store.exists(key)).toBe(true);
            }

            await store.deleteMany(keys);

            for (const key of keys) {
                expect(await store.exists(key)).toBe(false);
            }
        });
    });

    describe('list', () => {
        beforeEach(async () => {
            await store.upload('datasets/train.json', Buffer.from('train'));
            await store.upload('datasets/val.json', Buffer.from('val'));
            await store.upload('datasets/test.json', Buffer.from('test'));
            await store.upload('models/v1.bin', Buffer.from('model'));
            await store.upload('logs/2024-01-01.log', Buffer.from('log'));
        });

        it('should list files with prefix', async () => {
            const result = await store.list('datasets/');

            expect(result.keys).toHaveLength(3);
            expect(result.keys).toContain('datasets/train.json');
            expect(result.keys).toContain('datasets/val.json');
            expect(result.keys).toContain('datasets/test.json');
        });

        it('should list files with different prefix', async () => {
            const result = await store.list('models/');

            expect(result.keys).toHaveLength(1);
            expect(result.keys[0]).toBe('models/v1.bin');
        });

        it('should return empty for non-matching prefix', async () => {
            const result = await store.list('nonexistent/');

            expect(result.keys).toHaveLength(0);
        });

        it('should respect maxKeys option', async () => {
            const result = await store.list('datasets/', { maxKeys: 2 });

            expect(result.keys.length).toBeLessThanOrEqual(2);
        });
    });

    describe('getSignedUrl', () => {
        it('should generate a signed URL for download', async () => {
            const key = 'test/signed.txt';
            await store.upload(key, Buffer.from('signed content'));

            const url = await store.getSignedUrl(key, 3600);

            expect(url).toContain('test-bucket');
            expect(url).toContain(key);
            expect(url).toContain('expires=');
        });

        it('should generate a signed URL for upload', async () => {
            const key = 'test/upload-target.txt';

            const url = await store.getSignedUrl(key, 3600, 'put');

            expect(url).toContain(key);
        });
    });

    describe('exists', () => {
        it('should return true for existing file', async () => {
            const key = 'test/exists.txt';
            await store.upload(key, Buffer.from('exists'));

            expect(await store.exists(key)).toBe(true);
        });

        it('should return false for non-existent file', async () => {
            expect(await store.exists('non-existent-key')).toBe(false);
        });
    });

    describe('getMetadata', () => {
        it('should return file metadata', async () => {
            const content = 'metadata test content';
            const key = 'test/metadata.txt';

            await store.upload(key, Buffer.from(content), {
                contentType: 'text/plain',
            });

            const metadata = await store.getMetadata(key);

            expect(metadata.key).toBe(key);
            expect(metadata.size).toBe(content.length);
            expect(metadata.lastModified).toBeInstanceOf(Date);
            expect(metadata.contentType).toBe('text/plain');
            expect(metadata.etag).toBeDefined();
        });

        it('should throw for non-existent file', async () => {
            await expect(store.getMetadata('non-existent')).rejects.toThrow(
                'Object not found: non-existent'
            );
        });
    });

    describe('copy', () => {
        it('should copy a file to new location', async () => {
            const sourceKey = 'test/source.txt';
            const destKey = 'test/destination.txt';
            const content = 'copy test content';

            await store.upload(sourceKey, Buffer.from(content));
            await store.copy(sourceKey, destKey);

            expect(await store.exists(destKey)).toBe(true);

            const downloaded = await store.download(destKey);
            expect(downloaded.toString()).toBe(content);
        });

        it('should throw for non-existent source', async () => {
            await expect(store.copy('non-existent', 'dest')).rejects.toThrow(
                'Object not found: non-existent'
            );
        });
    });
});

describe('Utility Functions', () => {
    describe('generateUniqueKey', () => {
        it('should generate unique keys', () => {
            const key1 = generateUniqueKey('datasets');
            const key2 = generateUniqueKey('datasets');

            expect(key1).not.toBe(key2);
            expect(key1).toMatch(/^datasets\/\d+-[a-f0-9]+$/);
        });

        it('should include extension when provided', () => {
            const key = generateUniqueKey('models', 'bin');

            expect(key).toMatch(/^models\/\d+-[a-f0-9]+\.bin$/);
        });

        it('should handle extension with leading dot', () => {
            const key = generateUniqueKey('logs', '.log');

            expect(key).toMatch(/^logs\/\d+-[a-f0-9]+\.log$/);
        });
    });

    describe('parseStorageUri', () => {
        it('should parse S3 URI', () => {
            const result = parseStorageUri('s3://my-bucket/path/to/file.txt');

            expect(result.provider).toBe('s3');
            expect(result.bucket).toBe('my-bucket');
            expect(result.key).toBe('path/to/file.txt');
        });

        it('should parse GCS URI', () => {
            const result = parseStorageUri('gs://gcs-bucket/data/file.json');

            expect(result.provider).toBe('gcs');
            expect(result.bucket).toBe('gcs-bucket');
            expect(result.key).toBe('data/file.json');
        });

        it('should parse memory URI', () => {
            const result = parseStorageUri('memory://test-bucket/test/file.txt');

            expect(result.provider).toBe('s3'); // memory maps to s3
            expect(result.bucket).toBe('test-bucket');
            expect(result.key).toBe('test/file.txt');
        });

        it('should throw for invalid URI', () => {
            expect(() => parseStorageUri('invalid-uri')).toThrow('Invalid storage URI');
            expect(() => parseStorageUri('http://example.com')).toThrow('Invalid storage URI');
        });
    });

    describe('getCategoryKey', () => {
        it('should generate category-based keys', () => {
            expect(getCategoryKey('DATASETS', 'train.json')).toBe('datasets/train.json');
            expect(getCategoryKey('MODELS', 'v1.bin')).toBe('models/v1.bin');
            expect(getCategoryKey('LOGS', '2024-01-01.log')).toBe('logs/2024-01-01.log');
            expect(getCategoryKey('REPORTS', 'analysis.pdf')).toBe('reports/analysis.pdf');
        });
    });

    describe('StoragePaths', () => {
        it('should have all required paths', () => {
            expect(StoragePaths.DATASETS).toBe('datasets');
            expect(StoragePaths.MODELS).toBe('models');
            expect(StoragePaths.CHECKPOINTS).toBe('checkpoints');
            expect(StoragePaths.LOGS).toBe('logs');
            expect(StoragePaths.REPORTS).toBe('reports');
            expect(StoragePaths.EXPORTS).toBe('exports');
            expect(StoragePaths.TEMP).toBe('temp');
        });
    });
});

describe('Integration Scenarios', () => {
    let store: ReturnType<typeof createInMemoryStore>;

    beforeEach(() => {
        store = createInMemoryStore('prompt-refiner');
    });

    afterEach(() => {
        store.clear();
    });

    it('should handle training dataset workflow', async () => {
        // Upload training data
        const trainData = JSON.stringify({ examples: [1, 2, 3] });
        const trainKey = getCategoryKey('DATASETS', 'train-v1.json');

        await store.upload(trainKey, Buffer.from(trainData), {
            contentType: 'application/json',
            metadata: { version: '1.0', type: 'training' },
        });

        // Verify upload
        expect(await store.exists(trainKey)).toBe(true);

        // Download and verify
        const downloaded = await store.download(trainKey);
        expect(JSON.parse(downloaded.toString())).toEqual({ examples: [1, 2, 3] });

        // List all datasets
        const datasets = await store.list(StoragePaths.DATASETS + '/');
        expect(datasets.keys).toContain(trainKey);
    });

    it('should handle model checkpoint workflow', async () => {
        // Upload multiple checkpoints
        const checkpoints = [
            { key: getCategoryKey('CHECKPOINTS', 'epoch-1.pt'), content: 'checkpoint-1' },
            { key: getCategoryKey('CHECKPOINTS', 'epoch-2.pt'), content: 'checkpoint-2' },
            { key: getCategoryKey('CHECKPOINTS', 'epoch-3.pt'), content: 'checkpoint-3' },
        ];

        for (const cp of checkpoints) {
            await store.upload(cp.key, Buffer.from(cp.content));
        }

        // List all checkpoints
        const listed = await store.list(StoragePaths.CHECKPOINTS + '/');
        expect(listed.keys).toHaveLength(3);

        // Keep only latest checkpoint (cleanup old ones)
        await store.deleteMany([checkpoints[0].key, checkpoints[1].key]);

        const remaining = await store.list(StoragePaths.CHECKPOINTS + '/');
        expect(remaining.keys).toHaveLength(1);
        expect(remaining.keys[0]).toBe(checkpoints[2].key);
    });

    it('should handle report generation and sharing', async () => {
        // Generate report
        const reportContent = 'Benchmark Report: Score improved by 15%';
        const reportKey = generateUniqueKey(StoragePaths.REPORTS, 'txt');

        await store.upload(reportKey, Buffer.from(reportContent), {
            contentType: 'text/plain',
            metadata: { generatedBy: 'benchmark-system' },
        });

        // Generate signed URL for sharing (1 hour)
        const signedUrl = await store.getSignedUrl(reportKey, 3600);
        expect(signedUrl).toContain('expires=');

        // Verify metadata
        const metadata = await store.getMetadata(reportKey);
        expect(metadata.metadata?.generatedBy).toBe('benchmark-system');
    });
});
