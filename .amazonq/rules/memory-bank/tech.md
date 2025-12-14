# Prompt Architect - Technology Stack

## Programming Languages & Versions

### Primary Language
- **TypeScript 5.4.5**: Main development language for type safety and enhanced developer experience
- **JavaScript (ES5/ESNext)**: Target compilation for broad compatibility

### Frontend Framework
- **React 19.2.1**: Modern React with latest features for UI components
- **React DOM 19.2.1**: DOM rendering and interaction handling
- **JSX**: React component syntax with TypeScript integration

## Build System & Configuration

### TypeScript Configuration
**File**: `tsconfig.json`
**Key Settings**:
- Target: ES5 for compatibility
- Module: ESNext with Node resolution
- Strict mode enabled for type safety
- JSX: react-jsx for modern React
- Library support: DOM, DOM.iterable, ESNext

### Package Management
**File**: `package.json`
**Package Manager**: npm (with package-lock.json)
**Project Metadata**:
- Name: prompt-architect
- Version: 1.0.0
- Main Entry: prompt-engineer.tsx

## Dependencies

### Production Dependencies
```json
{
  "lucide-react": "^0.460.0",    // Icon library for UI components
  "react": "^19.2.1",           // Core React framework
  "react-dom": "^19.2.1"        // React DOM rendering
}
```

### Development Dependencies
```json
{
  "@types/react": "^19.0.0",        // React TypeScript definitions
  "@types/react-dom": "^19.0.0",    // React DOM TypeScript definitions
  "typescript": "^5.4.5"            // TypeScript compiler
}
```

## Development Commands

### Available Scripts
- `npm test`: Currently placeholder - no test framework configured yet
- Standard npm commands: `install`, `start`, `build` (to be configured)

### Recommended Development Workflow
```bash
# Install dependencies
npm install

# Type checking
npx tsc --noEmit

# Development server (to be configured)
# npm start

# Build for production (to be configured)  
# npm run build
```

## Architecture & Patterns

### Module System
- **ES Modules**: Modern import/export syntax
- **Node Resolution**: Standard Node.js module resolution
- **JSON Modules**: Support for importing JSON configuration files

### Type System
- **Strict TypeScript**: Full type checking enabled
- **Interface-Driven**: Strong typing for all data structures
- **Generic Types**: Flexible, reusable type definitions

### File Organization
- **Source Maps**: Enabled for debugging
- **Isolated Modules**: Each file treated as separate module
- **Consistent Casing**: Enforced file name casing

## Future Technology Considerations

### Planned Integrations
- **Testing Framework**: Jest or Vitest for unit testing
- **Build Tools**: Vite or Webpack for bundling
- **Linting**: ESLint with TypeScript rules
- **Formatting**: Prettier for code consistency

### AI/ML Integration
- **LLM APIs**: OpenAI, Anthropic, Groq for prompt evaluation
- **Vector Databases**: Pinecone/Weaviate for similarity search
- **Embedding Models**: For semantic similarity calculations

### Production Considerations
- **Bundling**: Module bundling for deployment
- **Environment Management**: Development/production configurations
- **Performance Monitoring**: Telemetry and analytics integration
- **Deployment**: CI/CD pipeline setup

## Development Environment

### IDE Support
- **Amazon Q Integration**: AI-powered development assistance
- **Claude AI Configuration**: Enhanced AI tooling support
- **TypeScript IntelliSense**: Full IDE support for type checking and auto-completion

### Code Quality
- **Type Safety**: Comprehensive TypeScript coverage
- **Modular Architecture**: Clean separation of concerns
- **Documentation**: Inline documentation and external guides