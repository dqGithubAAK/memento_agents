# Memento Framework: Educational Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts](#core-concepts)
3. [Architecture Overview](#architecture-overview)
4. [Memory System](#memory-system)
5. [Training Workflow](#training-workflow)
6. [Usage Examples](#usage-examples)
7. [Framework Comparison](#framework-comparison)
8. [Best Practices](#best-practices)

---

## Introduction

### What is Memento?

**Memento** is a framework for building LLM agents that learn from experience without fine-tuning the underlying language models. Instead of updating model weights, Memento uses **case-based reasoning (CBR)** and **parametric memory retrieval** to store and retrieve past experiences.

**Key Innovation**: "Fine-tuning LLM Agents without Fine-tuning LLMs"

### Why Memento?

Traditional approaches to improving LLM agents involve:
- ❌ **Fine-tuning**: Expensive, requires GPUs, risk of catastrophic forgetting
- ❌ **Prompt engineering**: Limited by context window, doesn't scale
- ❌ **Few-shot learning**: Manual example selection, no learning mechanism

Memento provides:
- ✅ **Memory-based learning**: Store successful/failed executions
- ✅ **Automatic retrieval**: Neural network selects relevant cases
- ✅ **Continual learning**: Improves over time without weight updates
- ✅ **Zero-shot generalization**: Works on new tasks out-of-the-box

### Performance

On the GAIA benchmark (complex real-world tasks):
- **87.88%** on validation set (state-of-the-art)
- **79.40%** on test set
- Outperforms GPT-4o, Claude 3.5 Sonnet, and other baselines

---

## Core Concepts

### 1. Case-Based Reasoning (CBR)

CBR is a problem-solving paradigm that uses past experiences (cases) to solve new problems.

**The CBR Cycle:**
```
1. Retrieve: Find similar past cases
2. Reuse: Adapt solutions from past cases
3. Revise: Test and refine the solution
4. Retain: Store new experience for future use
```

**Example:**
```
New Task: "Find all PDF files created last week"

Retrieved Case:
  Task: "Find all image files modified yesterday"
  Solution: Used `find` command with `-mtime` and `-type` flags
  Success: True

Adapted Solution:
  Use `find . -name "*.pdf" -mtime -7`
```

### 2. Parametric Memory

Unlike simple keyword matching, Memento uses a **trained neural network** to retrieve relevant cases.

**Components:**
- **Encoder**: Embeds queries and cases into vector space
- **Classifier**: Scores query-case pairs for relevance
- **Retriever**: Returns top-k most relevant cases

**Advantages:**
- Learns semantic similarity (not just keyword overlap)
- Improves over time as more data is collected
- Handles complex multi-step tasks

### 3. Hierarchical Planner-Executor

Memento uses a two-level architecture:

```
┌─────────────────────────────────────┐
│         Meta-Planner (GPT-4)        │
│   - Decomposes complex tasks        │
│   - Assigns subtasks to executor    │
│   - Manages overall workflow        │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│       Executor (o3/o4-mini)         │
│   - Executes individual subtasks    │
│   - Uses tools via MCP              │
│   - Returns results to planner      │
└─────────────────────────────────────┘
```

**Why two levels?**
- **Planner**: High-level reasoning (expensive, less frequent)
- **Executor**: Tool usage and execution (cheaper, more frequent)
- **Efficiency**: Only use expensive models when necessary

---

## Architecture Overview

### System Components

```
┌────────────────────────────────────────────────────┐
│                   User Query                        │
└──────────────────┬─────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│              Memory Retriever                       │
│  - Embeds query                                     │
│  - Retrieves top-8 similar cases                    │
│  - Returns cases with plans                         │
└──────────────────┬─────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│              Meta-Planner (GPT-4.1)                │
│  - Receives query + retrieved cases                 │
│  - Decomposes into subtasks                         │
│  - Plans execution strategy                         │
└──────────────────┬─────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│           Executor (o3/o4-mini)                    │
│  - Executes each subtask                            │
│  - Calls tools via MCP                              │
│  - Returns results                                  │
└──────────────────┬─────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│            Experience Logging                       │
│  - Stores: query, plan, trace, result               │
│  - LLM judge evaluates correctness                  │
│  - Saves to training dataset                        │
└──────────────────┬─────────────────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────────────────┐
│           Retriever Training                        │
│  - Trains on collected data                         │
│  - Updates retrieval policy                         │
│  - Improves future retrievals                       │
└────────────────────────────────────────────────────┘
```

### Data Flow

1. **Query Input**: User provides task description
2. **Memory Retrieval**: Retrieve K=8 similar past cases
3. **Planning**: Meta-planner creates execution plan (augmented with cases)
4. **Execution**: Executor runs plan, uses tools via MCP
5. **Logging**: Store full trace (query, plan, actions, result)
6. **Evaluation**: LLM judge determines success/failure
7. **Training**: Periodically update retriever on collected data

---

## Memory System

### Case Structure

Each case contains:

```python
{
    "task_id": "uuid-1234",
    "query": "Find the population of Tokyo",
    "plan": {
        "steps": [
            {"action": "search", "params": {"query": "Tokyo population"}},
            {"action": "extract", "params": {"field": "population"}}
        ]
    },
    "trace": {
        "meta_planner": [...],
        "executor": [...],
        "tool_calls": [...]
    },
    "result": "13.96 million",
    "label": "positive",  # or "negative"
    "metadata": {
        "tools_used": ["search", "extract"],
        "execution_time": 5.2,
        "timestamp": "2024-..."
    }
}
```

### Neural Retriever

**Architecture:**
```
Input: [QUERY] user query [CASE] case query [PLAN] case plan

┌──────────────────────────┐
│   Sentence Encoder        │
│  (sup-simcse-roberta)     │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  [CLS] embedding          │
│  (768-dim vector)         │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Linear + Dropout         │
│  768 → 2                  │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Softmax                  │
│  [irrelevant, relevant]   │
└──────────────────────────┘
```

**Training:**
- **Positive examples**: Cases that led to successful task completion
- **Negative examples**: Cases that led to failures or are irrelevant
- **Loss**: Cross-entropy on binary classification
- **Optimization**: AdamW with learning rate 2e-5

### Retrieval Process

```python
def retrieve(query: str, case_pool: List[Case], k: int = 8) -> List[Case]:
    """
    Retrieve top-k most relevant cases for a query.
    """
    # 1. Format query and cases
    query_text = f"[QUERY] {query}"
    
    # 2. Score all cases
    scores = []
    for case in case_pool:
        case_text = f"[CASE] {case.query} [PLAN] {case.plan}"
        input_text = f"{query_text} {case_text}"
        
        # Forward pass through neural network
        score = model.predict_relevance(input_text)
        scores.append((score, case))
    
    # 3. Sort by score and return top-k
    scores.sort(reverse=True, key=lambda x: x[0])
    top_cases = [case for score, case in scores[:k]]
    
    return top_cases
```

---

## Training Workflow

### Phase 1: Data Collection

Run agent on tasks and automatically collect training data.

```python
from Memento.client.parametric_memory_cbr import HierarchicalClientWithMemory

# Initialize agent
agent = HierarchicalClientWithMemory(
    memory_retriever=retriever,
    mcp_servers={"search": SearchServer(), "filesystem": FileSystemServer()}
)

# Run on tasks
for task in tasks:
    result = agent.run(task)
    
    # Automatically logs:
    # - Query
    # - Retrieved cases
    # - Generated plan
    # - Execution trace
    # - Tool calls
    # - Final result
```

### Phase 2: Reward Assignment

Use an LLM judge to evaluate correctness.

```python
def evaluate_result(task: Task, result: str, ground_truth: str) -> bool:
    """
    Use GPT-4 to judge if result is correct.
    """
    prompt = f"""
Task: {task.description}
Expected Answer: {ground_truth}
Agent's Answer: {result}

Is the agent's answer correct? Answer with "yes" or "no".
"""
    
    judgment = gpt4(prompt)
    return "yes" in judgment.lower()

# Label cases
for record in execution_records:
    is_correct = evaluate_result(
        record.task,
        record.result,
        record.ground_truth
    )
    
    record.label = "positive" if is_correct else "negative"
    save_to_dataset(record)
```

### Phase 3: Retriever Training

Train the neural retriever on collected data.

```python
from Memento.memory.parametric_memory import CaseRetriever

# Load training data
train_data = load_training_data("memory.jsonl")

# Prepare pairs
positive_pairs = []
negative_pairs = []

for record in train_data:
    query = record.query
    
    if record.label == "positive":
        # Query should retrieve this case
        positive_pairs.append((query, record.to_case()))
    else:
        # Query should NOT retrieve this case
        negative_pairs.append((query, record.to_case()))

# Train retriever
retriever = CaseRetriever()
retriever.train(
    positive_pairs=positive_pairs,
    negative_pairs=negative_pairs,
    epochs=10,
    batch_size=16,
    learning_rate=2e-5
)

# Save trained model
retriever.save("models/retriever_v2.pt")
```

### Phase 4: Evaluation

Test on held-out validation set.

```python
# Load trained retriever
retriever = CaseRetriever.load("models/retriever_v2.pt")

# Evaluate retrieval quality
for query, relevant_cases in validation_set:
    retrieved = retriever.retrieve(query, case_pool, k=8)
    
    # Calculate metrics
    precision = len(set(retrieved) & set(relevant_cases)) / len(retrieved)
    recall = len(set(retrieved) & set(relevant_cases)) / len(relevant_cases)
    
    print(f"Precision: {precision:.2%}, Recall: {recall:.2%}")
```

---

## Usage Examples

### Example 1: Simple Agent Setup

```python
from Memento.client.agent import HierarchicalClient
from Memento.client.mcp_toolkit import MCPToolkit

# 1. Define MCP tools
toolkit = MCPToolkit()
toolkit.add_server("search", "mcp-server-brave-search")
toolkit.add_server("filesystem", "mcp-server-filesystem")

# 2. Initialize agent
agent = HierarchicalClient(
    planner_model="gpt-4-turbo",
    executor_model="o3-mini",
    mcp_toolkit=toolkit
)

# 3. Run a task
result = agent.run("Find all Python files in this directory")

print(f"Result: {result.answer}")
print(f"Steps: {len(result.trace)}")
```

### Example 2: Agent with Memory

```python
from Memento.client.parametric_memory_cbr import HierarchicalClientWithMemory
from Memento.memory.parametric_memory import CaseRetriever

# 1. Load trained retriever
retriever = CaseRetriever.load("models/retriever.pt")

# 2. Initialize memory-augmented agent
agent = HierarchicalClientWithMemory(
    planner_model="gpt-4-turbo",
    executor_model="o3-mini",
    memory_retriever=retriever,
    memory_top_k=8,  # Retrieve 8 cases
    mcp_toolkit=toolkit
)

# 3. Run with memory
result = agent.run("Calculate the average temperature in Tokyo last week")

# 4. Inspect retrieved cases
print(f"Retrieved {len(result.retrieved_cases)} relevant cases:")
for i, case in enumerate(result.retrieved_cases, 1):
    print(f"{i}. {case.query} (score={case.score:.3f})")
```

### Example 3: Training Loop

```python
from Memento.training.data_collector import DataCollector
from Memento.training.trainer import RetrieverTrainer

# 1. Initialize data collector
collector = DataCollector(
    agent=agent,
    output_path="training_data/memory.jsonl"
)

# 2. Collect data on tasks
for task in training_tasks:
    result = agent.run(task.query)
    
    # Automatically evaluates and logs
    collector.log_execution(
        query=task.query,
        result=result,
        ground_truth=task.answer
    )

print(f"Collected {collector.num_records} training examples")

# 3. Train retriever
trainer = RetrieverTrainer(
    training_data="training_data/memory.jsonl",
    model_path="models/retriever_v2.pt"
)

trainer.train(
    epochs=10,
    batch_size=16,
    learning_rate=2e-5,
    validation_split=0.2
)

# 4. Evaluate
metrics = trainer.evaluate(test_set)
print(f"Validation Accuracy: {metrics['accuracy']:.2%}")
print(f"Retrieval Precision@8: {metrics['precision@8']:.2%}")
```

---

## Framework Comparison

### LangGraph vs Memento vs Custom

| Aspect | **LangGraph** | **Memento** | **Custom Implementation** |
|--------|--------------|-------------|--------------------------|
| **Architecture** | Graph-based workflows | Hierarchical planner-executor | Fully custom |
| **Memory** | Built-in checkpointing | Parametric case-based reasoning | Manual implementation |
| **Learning** | No native learning | Automatic from experience | Custom RL/training |
| **Deployment** | Production-ready (v0.6.5) | Research framework | Depends on implementation |
| **Ease of Use** | High (prebuilt agents) | Medium (requires training) | Low (build from scratch) |
| **Flexibility** | High (custom graphs) | Medium (fixed architecture) | Very high |
| **Tool Integration** | Native LangChain tools | MCP protocol | Manual tool wrapping |
| **Multi-agent** | Built-in support | Not native | Custom coordination |
| **Human-in-loop** | Built-in | Not native | Custom implementation |
| **Best For** | Production systems | Research, continual learning | Custom research needs |

### When to Use Each

**Use LangGraph if:**
- ✅ Building production-ready systems
- ✅ Need human-in-the-loop workflows
- ✅ Want pre-built agent templates
- ✅ Require durable execution and checkpointing
- ✅ Working with LangChain ecosystem

**Use Memento if:**
- ✅ Researching agent learning and adaptation
- ✅ Want automatic improvement from experience
- ✅ Need case-based reasoning
- ✅ Don't want to fine-tune LLMs
- ✅ Have evaluation benchmark with clear success/failure

**Build Custom if:**
- ✅ Have specific research requirements
- ✅ Need full control over architecture
- ✅ Implementing novel algorithms
- ✅ Have team with strong ML engineering
- ✅ Want maximum flexibility

### Hybrid Approach: LangGraph + Memento Concepts

You can combine the best of both:

```python
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import Tool

class MemoryAugmentedLangGraphAgent:
    """
    LangGraph agent with Memento-style memory.
    """
    def __init__(self, llm, tools, memory_retriever):
        self.memory_retriever = memory_retriever
        self.agent = create_react_agent(llm, tools)
    
    def run(self, query: str):
        # Retrieve relevant cases
        cases = self.memory_retriever.retrieve(query, k=5)
        
        # Augment query with memory context
        memory_context = self._format_cases(cases)
        augmented_query = f"{memory_context}\n\nTask: {query}"
        
        # Run LangGraph agent
        result = self.agent.invoke({"messages": [("user", augmented_query)]})
        
        # Store experience
        self._store_experience(query, result)
        
        return result
```

**Advantages:**
- 🎯 LangGraph's production-ready infrastructure
- 🧠 Memento's learning from experience
- 🔧 Easy tool integration
- 📊 Automatic memory improvement

---

## Best Practices

### 1. Memory Management

**Do:**
- ✅ Store complete execution traces (not just final answers)
- ✅ Include tool usage patterns in cases
- ✅ Track execution time and resource usage
- ✅ Version your memory store and retriever models

**Don't:**
- ❌ Store redundant or duplicate cases
- ❌ Keep cases without clear success/failure labels
- ❌ Ignore memory pruning (remove outdated cases)
- ❌ Mix cases from different task distributions

### 2. Retrieval Strategy

**Do:**
- ✅ Use K=5-10 cases for most tasks
- ✅ Filter by success rate (e.g., only retrieve successful cases)
- ✅ Diversify retrieved cases (avoid too similar examples)
- ✅ Re-rank cases by execution time or complexity

**Don't:**
- ❌ Retrieve too many cases (context overflow)
- ❌ Use fixed retrieval without adaptation
- ❌ Ignore case metadata (tools, tags, etc.)
- ❌ Retrieve cases from very different task types

### 3. Training

**Do:**
- ✅ Collect balanced positive/negative examples
- ✅ Use held-out validation set for evaluation
- ✅ Monitor retrieval quality metrics (P@K, R@K)
- ✅ Retrain periodically as new data arrives

**Don't:**
- ❌ Overtrain on small datasets
- ❌ Ignore class imbalance (success vs failure ratio)
- ❌ Use biased LLM judge (ensure fair evaluation)
- ❌ Train without validation

### 4. Integration

**Do:**
- ✅ Start with simple agent architecture
- ✅ Gradually add memory capabilities
- ✅ Monitor memory impact on performance
- ✅ A/B test with and without memory

**Don't:**
- ❌ Add memory without clear evaluation
- ❌ Expect immediate improvements (requires data)
- ❌ Ignore memory overhead (latency, storage)
- ❌ Over-complicate the architecture

---

## Summary

**Memento** provides a powerful framework for building LLM agents that learn from experience:

1. **Case-Based Reasoning**: Store and retrieve past executions
2. **Parametric Memory**: Train neural retriever for semantic search
3. **Hierarchical Architecture**: Separate planning and execution
4. **Automatic Training**: Learn from task outcomes without manual labeling
5. **Continual Learning**: Improve over time without fine-tuning LLMs

**Key Advantages:**
- 🚀 State-of-the-art performance on GAIA benchmark
- 🧠 Learn from experience automatically
- 💰 No expensive fine-tuning required
- 🔄 Continually improve with usage
- 🎯 Maintain base LLM capabilities

**Next Steps:**
1. Read the [Integration Guide](INTEGRATION_ARE_MEMENTO.md) for combining ARE + Memento
2. Try the [Memento Tutorial Notebook](notebooks/memento_tutorial.ipynb)
3. Explore the [Memento repository](https://github.com/Agent-on-the-Fly/Memento) for more details

---

## References

- **Memento Paper**: "Fine-tuning LLM Agents without Fine-tuning LLMs"
- **GAIA Benchmark**: https://huggingface.co/gaia-benchmark
- **MCP Protocol**: Model Context Protocol for tool integration
- **LangGraph**: https://langchain-ai.github.io/langgraph/
