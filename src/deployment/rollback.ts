
/**
 * Rollback and Version Control System (DIRECTIVE-032)
 * Handles snapshotting prompts and reverting to previous versions.
 */

export interface PromptVersion {
    versionId: string;
    promptId: string;
    content: string;
    timestamp: Date;
    author: string;
    metadata?: Record<string, any>;
    commitMessage?: string;
}

// In-memory version store (replace with Database in production)
const versionStore: Map<string, PromptVersion[]> = new Map();

/**
 * Save a snapshot of a prompt (Create new version)
 */
export function createSnapshot(
    promptId: string,
    content: string,
    author: string = 'system',
    message: string = 'Auto-snapshot'
): PromptVersion {
    const versionId = `v_${Date.now()}`;
    const version: PromptVersion = {
        versionId,
        promptId,
        content,
        timestamp: new Date(),
        author,
        commitMessage: message
    };

    const versions = versionStore.get(promptId) || [];
    versions.push(version);
    versionStore.set(promptId, versions);

    return version;
}

/**
 * Get full version history for a prompt
 */
export function getVersionHistory(promptId: string): PromptVersion[] {
    return (versionStore.get(promptId) || []).sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
}

/**
 * Revert to a specific version
 * This essentially creates a NEW version that is a copy of the OLD version,
 * to preserve linear history.
 */
export function rollback(promptId: string, targetVersionId: string): PromptVersion | null {
    const versions = versionStore.get(promptId);
    if (!versions) return null;

    const targetVersion = versions.find(v => v.versionId === targetVersionId);
    if (!targetVersion) return null;

    // Create a new snapshot based on the old one
    return createSnapshot(
        promptId,
        targetVersion.content,
        'system',
        `Rollback to version ${targetVersionId}`
    );
}

/**
 * Get the latest version of a prompt
 */
export function getLatestVersion(promptId: string): PromptVersion | null {
    const history = getVersionHistory(promptId);
    return history.length > 0 ? history[0] : null;
}
