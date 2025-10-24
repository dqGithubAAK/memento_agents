# GAIA2 + Memento: Continual Learning for LLM Agents

This repository contains comprehensive documentation and implementation guides for building continually learning LLM agents by combining **ARE (Agents Research Environments)** with **Memento's case-based reasoning framework**.

## 📚 Overview

**Goal**: Build agents that improve their performance over time by learning from experience in dynamic evaluation environments.

**Key Innovation**: Combine ARE's rich evaluation infrastructure with Memento's memory-augmented learning to create agents that:
- Execute complex tasks in realistic simulations
- Store successful/failed strategies in memory
- Retrieve relevant past experiences
- Train retrieval policies to select better memories
- Continually improve without fine-tuning base LLMs

## 🎯 What's Included

### Documentation

1. **[CUSTOM_AGENT_GUIDE.md](ARE/CUSTOM_AGENT_GUIDE.md)**
   - Comprehensive guide to implementing custom agents on ARE
   - LLM integration patterns
   - Tool usage and notification handling
   - Task execution and validation
   - Memory integration strategies

2. **[MEMENTO_GUIDE.md](MEMENTO_GUIDE.md)**
   - Educational guide to Memento framework
   - Case-based reasoning concepts
   - Parametric memory architecture
   - Training workflow
   - Framework comparison (LangGraph, Memento, custom)
   - Best practices and usage examples

3. **[INTEGRATION_ARE_MEMENTO.md](INTEGRATION_ARE_MEMENTO.md)**
   - Complete architecture for ARE + Memento integration
   - System design and data flow
   - Reward derivation strategies
   - Training pipeline
   - Deployment architecture
   - Bottleneck analysis and solutions
   - Implementation roadmap

### Interactive Notebooks

#### ARE Framework Tutorials

1. **[01_understanding_are_framework.ipynb](ARE/notebooks/01_understanding_are_framework.ipynb)**
   - Introduction to ARE's event-driven architecture
   - Loading and inspecting scenarios
   - Exploring apps, tools, and event DAGs
   - Running oracle mode for ground truth
   - Analyzing execution traces

2. **[02_running_custom_agent.ipynb](ARE/notebooks/02_running_custom_agent.ipynb)**
   - Implementing RunnableARESimulationAgent interface
   - Building LangGraph agent for ARE
   - Tool conversion and integration
   - Notification system handling
   - Testing on real scenarios

3. **[03_agent_with_memory.ipynb](ARE/notebooks/03_agent_with_memory.ipynb)**
   - Memory structures (episodic and semantic)
   - Vector-based memory stores (ChromaDB)
   - Memory-augmented agent implementation
   - Learning from successes and failures
   - Pattern analysis and strategy extraction

#### Memento Framework Tutorial

4. **[memento_tutorial.ipynb](notebooks/memento_tutorial.ipynb)**
   - Case-based reasoning fundamentals
   - Semantic retrieval with embeddings
   - Building memory-augmented agents
   - Parametric memory and neural retrievers
   - Framework comparison (LangGraph vs Memento vs custom)
   - Integration patterns

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+ required (3.12 for ARE)
python --version

# Install ARE
cd ARE/meta-agents-research-environments
pip install -e .

# Install Memento dependencies
pip install torch transformers sentence-transformers chromadb
```

### Run Your First Agent

```python
from are.simulation.environment import Environment, EnvironmentConfig
from are.simulation.scenarios.registration import get_scenario_by_id

# Load a scenario
scenario = get_scenario_by_id("scenario_tutorial")
scenario.initialize()

# Create your agent (see notebooks for details)
from my_agent import MyCustomAgent
agent = MyCustomAgent()

# Run
result = agent.run_scenario(scenario, notification_system=None)
print(f"Success: {result.success}")
```

### Start Learning from Experience

```python
from memory_store import MemoryStore
from retriever import CaseRetriever

# Initialize memory
memory = MemoryStore("memory.jsonl")
retriever = CaseRetriever()

# Run agent with memory
from memory_agent import MemoryAugmentedAgent
agent = MemoryAugmentedAgent(retriever, memory)

# Execute and learn
for scenario in scenarios:
    result = agent.run_scenario(scenario)
    # Automatically stores experience and trains retriever
```

## 📖 Learning Path

### For Beginners

1. Start with **[01_understanding_are_framework.ipynb](ARE/notebooks/01_understanding_are_framework.ipynb)**
   - Understand the evaluation environment
   - Learn about scenarios, apps, and tools

2. Read **[CUSTOM_AGENT_GUIDE.md](ARE/CUSTOM_AGENT_GUIDE.md)**
   - Learn how to build custom agents
   - Understand LLM integration patterns

3. Try **[02_running_custom_agent.ipynb](ARE/notebooks/02_running_custom_agent.ipynb)**
   - Implement your first LangGraph agent
   - Test on real scenarios

### For Intermediate Users

4. Read **[MEMENTO_GUIDE.md](MEMENTO_GUIDE.md)**
   - Understand case-based reasoning
   - Learn about parametric memory

5. Work through **[memento_tutorial.ipynb](notebooks/memento_tutorial.ipynb)**
   - Build semantic retrieval systems
   - Train neural retrievers

6. Explore **[03_agent_with_memory.ipynb](ARE/notebooks/03_agent_with_memory.ipynb)**
   - Implement memory-augmented agents
   - Analyze learning patterns

### For Advanced Users

7. Study **[INTEGRATION_ARE_MEMENTO.md](INTEGRATION_ARE_MEMENTO.md)**
   - Understand full integration architecture
   - Design production systems
   - Optimize for performance

8. Implement your own continual learning system
   - Combine ARE + Memento concepts
   - Train on GAIA2 benchmark
   - Publish results!

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              ARE Simulation Environment                  │
│  - 800 GAIA2 scenarios across 7 capabilities            │
│  - 600+ tools in 7 apps (search, email, files, etc.)    │
│  - Dynamic event-driven execution                        │
│  - Ground truth validation                               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          Memory-Augmented Agent (Your Code)              │
│  1. Retrieve relevant cases from memory                 │
│  2. Plan using retrieved examples                       │
│  3. Execute using ARE tools                             │
│  4. Store experience with reward                        │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Memento Memory System                       │
│  - Case store (episodes + rewards)                      │
│  - Neural retriever (trained on outcomes)               │
│  - Continual learning pipeline                          │
│  - No base LLM fine-tuning required                     │
└─────────────────────────────────────────────────────────┘
```

## 🎓 Key Concepts

### ARE (Agents Research Environments)

- **Purpose**: Realistic evaluation of LLM agents
- **Scenarios**: Complex multi-step tasks requiring planning, tool use, and adaptation
- **Apps**: Search, email, calendar, files, notifications, web browser, multimedia
- **Validation**: Task-specific verifiers provide clear success/failure signals
- **Benchmark**: GAIA2 with 800 scenarios testing 7 key capabilities

### Memento (Case-Based Reasoning)

- **Purpose**: "Fine-tuning LLM agents without fine-tuning LLMs"
- **Memory**: Store complete execution traces with outcomes
- **Retrieval**: Neural network learns to select relevant cases
- **Learning**: Train retriever on collected experience data
- **Performance**: 87.88% on GAIA validation, 79.40% on test

### Integration Benefits

✅ **Dynamic Evaluation**: Test on complex, realistic scenarios  
✅ **Clear Rewards**: ARE validation provides training signal  
✅ **Rich Context**: Store tool usage, plans, and outcomes  
✅ **Continual Learning**: Improve over time without model updates  
✅ **Scalable**: Add more memories without retraining base LLM  

## 📊 Performance Expectations

Based on Memento's results on GAIA:

| Metric | Baseline (GPT-4) | With Memory (Expected) |
|--------|------------------|------------------------|
| Success Rate | ~60-70% | ~75-85% |
| Avg. Steps | 8-12 | 6-10 |
| Retrieval Benefit | N/A | +10-20% |

**Timeline to see improvements:**
- After 50 scenarios: Minimal (cold start)
- After 100 scenarios: 5-10% improvement
- After 500 scenarios: 10-20% improvement
- After 1000+ scenarios: Plateau at ~15-25% improvement

## 🔬 Research Directions

### Unexplored Areas

1. **Cross-Task Transfer**: Do memories from task A help with task B?
2. **Memory Compression**: Can we distill 1000 cases into general strategies?
3. **Active Learning**: Which scenarios to prioritize for maximum learning?
4. **Multi-Agent Memory**: Share memories across agent populations?
5. **Hierarchical Memory**: Organize memories by abstraction level?

### Open Questions

- What's the optimal memory retrieval strategy (K=5? K=10? dynamic?)?
- How to handle catastrophic forgetting as memory grows?
- Can we learn meta-strategies that generalize across domains?
- How to balance exploration (new strategies) vs exploitation (known solutions)?

## 🛠️ Tools and Technologies

### Core Frameworks

- **ARE**: Meta's agent evaluation framework
- **Memento**: Case-based reasoning for agents
- **LangGraph** (optional): Production-ready agent framework
- **MCP**: Model Context Protocol for tool integration

### Key Libraries

```
torch>=2.0.0                    # Neural network training
transformers>=4.30.0            # Pretrained models
sentence-transformers>=2.2.0    # Semantic embeddings
chromadb>=0.4.0                # Vector database
langgraph>=0.6.0               # Agent framework (optional)
openai>=1.0.0                  # LLM API
```

## 📝 Documentation Best Practices

This repository follows industry-standard documentation practices:

- ✅ **Progressive Disclosure**: Start simple, add complexity gradually
- ✅ **Code Examples**: Every concept illustrated with runnable code
- ✅ **Visual Aids**: Diagrams and architecture illustrations
- ✅ **Multiple Formats**: Markdown guides + Jupyter notebooks
- ✅ **Cross-References**: Links between related documents
- ✅ **Practical Focus**: Real-world usage patterns

## 🤝 Contributing

This is educational documentation. To contribute:

1. **Fix errors**: Submit PRs for corrections
2. **Add examples**: Share your implementations
3. **Improve clarity**: Suggest better explanations
4. **Extend coverage**: Add missing topics

## 📄 License

- **ARE**: Meta's license (see ARE/meta-agents-research-environments/)
- **Memento**: See Memento repository
- **This documentation**: MIT License

## 🔗 References

### Official Repositories

- **ARE**: https://github.com/facebookresearch/meta-agents-research-environments
- **Memento**: https://github.com/Agent-on-the-Fly/Memento
- **GAIA Benchmark**: https://huggingface.co/gaia-benchmark
- **LangGraph**: https://github.com/langchain-ai/langgraph

### Papers

- **GAIA**: "GAIA: A Benchmark for General AI Assistants"
- **Memento**: "Fine-tuning LLM Agents without Fine-tuning LLMs" (coming soon)
- **Case-Based Reasoning**: Classical AI literature

### Related Work

- **ReAct**: "Synergizing Reasoning and Acting in Language Models"
- **Toolformer**: "Language Models Can Teach Themselves to Use Tools"
- **Memory Networks**: "End-to-End Memory Networks"

## 📧 Contact

For questions about:
- **ARE**: Check Meta's repository issues
- **Memento**: Check Memento repository
- **This documentation**: Open an issue in this repository

## 🙏 Acknowledgments

- **Meta AI** for the ARE framework
- **Agent-on-the-Fly team** for Memento
- **GAIA team** for the benchmark
- **LangChain team** for LangGraph
- **OpenAI & Anthropic** for powerful LLMs

---

## 🚀 Get Started Now!

1. Clone this repository
2. Read [CUSTOM_AGENT_GUIDE.md](ARE/CUSTOM_AGENT_GUIDE.md)
3. Run [01_understanding_are_framework.ipynb](ARE/notebooks/01_understanding_are_framework.ipynb)
4. Build your first memory-augmented agent!

**Happy Learning! 🎉**
