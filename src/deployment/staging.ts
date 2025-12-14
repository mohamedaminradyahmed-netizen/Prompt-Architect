
/**
 * Staging Environment Management (DIRECTIVE-032)
 * Handles applying variations to a staging environment and A/B testing setup.
 */

export interface StagingDeployment {
    deploymentId: string;
    variationId: string;
    promptContent: string;
    deployedAt: Date;
    status: 'active' | 'inactive';
    abTestConfig?: ABTestConfig;
}

export interface ABTestConfig {
    trafficSplit: number; // 0.0 to 1.0 (e.g., 0.5 for 50/50)
    startDate: Date;
    durationHours: number;
}

// In-memory store for deployments (replace with DB in production)
const stagingDeployments: Map<string, StagingDeployment> = new Map();

/**
 * Deploy a variation to the staging environment
 */
export async function deployToStaging(
    variationId: string,
    promptContent: string,
    abTestConfig?: ABTestConfig
): Promise<StagingDeployment> {
    const deploymentId = `deploy_${Date.now()}_${Math.floor(Math.random() * 1000)}`;

    const deployment: StagingDeployment = {
        deploymentId,
        variationId,
        promptContent,
        deployedAt: new Date(),
        status: 'active',
        abTestConfig
    };

    stagingDeployments.set(deploymentId, deployment);

    // In a real system, this would trigger a config update or webhook
    console.log(`[Staging] Deployed variation ${variationId} to staging (ID: ${deploymentId})`);

    return deployment;
}

/**
 * Get active staging deployment
 */
export function getActiveStagingDeployment(): StagingDeployment | undefined {
    // Return the most recent active deployment
    const active = Array.from(stagingDeployments.values())
        .filter(d => d.status === 'active')
        .sort((a, b) => b.deployedAt.getTime() - a.deployedAt.getTime());

    return active[0];
}

/**
 * Deactivate a deployment
 */
export function deactivateDeployment(deploymentId: string): boolean {
    const deployment = stagingDeployments.get(deploymentId);
    if (deployment) {
        deployment.status = 'inactive';
        stagingDeployments.set(deploymentId, deployment);
        return true;
    }
    return false;
}
