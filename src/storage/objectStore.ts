import { S3Client, PutObjectCommand, GetObjectCommand, DeleteObjectCommand, ListObjectsV2Command } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { Readable } from 'stream';

export class ObjectStore {
  private client: S3Client;
  private bucket: string;

  constructor(bucket: string = process.env.STORAGE_BUCKET || 'prompt-refiner-storage', region: string = process.env.AWS_REGION || 'us-east-1') {
    this.client = new S3Client({ region });
    this.bucket = bucket;
  }

  async upload(key: string, data: Buffer | Readable | string | Uint8Array): Promise<string> {
    const command = new PutObjectCommand({
      Bucket: this.bucket,
      Key: key,
      Body: data,
    });

    await this.client.send(command);
    return `s3://${this.bucket}/${key}`;
  }

  async download(key: string): Promise<Buffer> {
    const command = new GetObjectCommand({
      Bucket: this.bucket,
      Key: key,
    });

    const response = await this.client.send(command);

    if (!response.Body) {
      throw new Error(`File not found or empty: ${key}`);
    }

    return this.streamToBuffer(response.Body as Readable);
  }

  async delete(key: string): Promise<void> {
    const command = new DeleteObjectCommand({
      Bucket: this.bucket,
      Key: key,
    });
    await this.client.send(command);
  }

  async list(prefix: string): Promise<string[]> {
    const command = new ListObjectsV2Command({
      Bucket: this.bucket,
      Prefix: prefix,
    });

    const response = await this.client.send(command);
    return response.Contents?.map(c => c.Key || '').filter(k => k) || [];
  }

  async getSignedUrl(key: string, expiresIn: number = 3600): Promise<string> {
    const command = new GetObjectCommand({
      Bucket: this.bucket,
      Key: key,
    });
    return getSignedUrl(this.client, command, { expiresIn });
  }

  private async streamToBuffer(stream: Readable | Blob | any): Promise<Buffer> {
    if (Buffer.isBuffer(stream)) {
        return stream;
    }

    if (typeof stream.on === 'function') {
        const chunks: Buffer[] = [];
        return new Promise((resolve, reject) => {
            stream.on('data', (chunk: any) => chunks.push(Buffer.from(chunk)));
            stream.on('error', (err: any) => reject(err));
            stream.on('end', () => resolve(Buffer.concat(chunks)));
        });
    }

    throw new Error('Unsupported stream type');
  }
}
