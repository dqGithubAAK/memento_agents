# ARE + Memento Integration: Continual Learning Architecture

## Executive Summary

This document outlines the architecture for integrating **ARE (Agents Research Environments)** with **Memento's case-based reasoning framework** to create agents that continuously learn from experience in dynamic evaluation environments.

**Goal**: Build agents that improve their performance over time by:
1. Executing tasks in ARE's dynamic simulation environment
2. Deriving rewards from task success/failure
3. Storing action sequences in Memento's memory system
4. Training a retrieval policy to select relevant past experiences
5. Using retrieved experiences to guide future task execution

**Expected Outcome**: Agents that achieve higher success rates on complex tasks through experience accumulation and adaptive strategy selection.

---

## Table of Contents

1. [Integration Overview](#integration-overview)
2. [Architecture Design](#architecture-design)
3. [Data Flow](#data-flow)
4. [Implementation Components](#implementation-components)
5. [Reward Derivation Strategy](#reward-derivation-strategy)
6. [Training Pipeline](#training-pipeline)
7. [Deployment Architecture](#deployment-architecture)
8. [Potential Bottlenecks](#potential-bottlenecks)
9. [Implementation Roadmap](#implementation-roadmap)
10. [Code Examples](#code-examples)

---

## Integration Overview

### Why Integrate ARE + Memento?

**ARE Provides:**
- ✅ Dynamic, event-driven evaluation environment
- ✅ Rich tool ecosystem (600+ tools across 7 apps)
- ✅ Complex multi-step scenarios (GAIA2 benchmark)
- ✅ Clear success/failure validation
- ✅ Execution traces and event notifications

**Memento Provides:**
- ✅ Case-based reasoning framework
- ✅ Parametric memory with neural retriever
- ✅ Automatic training data collection
- ✅ Learning from experience without fine-tuning
- ✅ Continual improvement mechanism

**Combined System:**
```
ARE (Evaluation) + Memento (Learning) = Continually Improving Agent
```

### High-Level Concept

```
┌─────────────────────────────────────────────────────────┐
│                     ARE Environment                      │
│  - Provides tasks and dynamic simulation                │
│  - Validates agent outputs                              │
│  - Generates reward signals                             │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│                Agent Execution Layer                     │
│  - Retrieves relevant past experiences                  │
│  - Plans and executes actions                           │
│  - Interacts with ARE tools                             │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│               Memento Memory System                      │
│  - Stores execution traces                              │
│  - Trains retrieval policy                              │
│  - Provides case-based guidance                         │
└─────────────────────────────────────────────────────────┘
```

---

## Architecture Design

### System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        ARE Scenario                          │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Event DAG: event_1 → event_2 → ... → event_n     │     │
│  │  Apps: Search, Email, Files, Calendar, etc.        │     │
│  │  Validation: Ground truth checker                  │     │
│  └────────────────────────────────────────────────────┘     │
└────────────────────────┬─────────────────────────────────────┘
                         │ 1. Task Description
                         │    + Available Tools
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                   Memory Retrieval Layer                     │
│  ┌────────────────────────────────────────────────────┐     │
│  │  1. Encode current task                            │     │
│  │  2. Query Memento's case database                  │     │
│  │  3. Retrieve K=5-10 similar cases                  │     │
│  │  4. Filter by success rate, tools, tags            │     │
│  └────────────────────────────────────────────────────┘     │
└────────────────────────┬─────────────────────────────────────┘
                         │ 2. Retrieved Cases
                         │    + Scores
                         ▼
┌──────────────────────────────────────────────────────────────┐
│              Hierarchical Agent (Planner + Executor)         │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Meta-Planner (GPT-4):                             │     │
│  │    - Receives task + retrieved cases                │     │
│  │    - Decomposes into subtasks                       │     │
│  │    - Generates execution plan                       │     │
│  │                                                      │     │
│  │  Executor (o3-mini):                                │     │
│  │    - Executes each subtask                          │     │
│  │    - Calls ARE tools via adapter                    │     │
│  │    - Returns results to planner                     │     │
│  └────────────────────────────────────────────────────┘     │
└────────────────────────┬─────────────────────────────────────┘
                         │ 3. Execution Trace
                         │    + Tool Calls
                         │    + Results
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                      ARE Validation                          │
│  ┌────────────────────────────────────────────────────┐     │
│  │  - Compare output to ground truth                   │     │
│  │  - Use task-specific verifier                       │     │
│  │  - Generate success/failure signal                  │     │
│  └────────────────────────────────────────────────────┘     │
└────────────────────────┬─────────────────────────────────────┘
                         │ 4. Validation Result
                         │    (Success/Failure)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    Reward Computation                        │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Binary Reward:                                     │     │
│  │    R = 1.0 if success, 0.0 if failure               │     │
│  │                                                      │     │
│  │  Shaped Reward (optional):                          │     │
│  │    R = α * success + β * efficiency + γ * quality   │     │
│  │                                                      │     │
│  │  Components:                                        │     │
│  │    - Success: Task completion                       │     │
│  │    - Efficiency: Steps used / optimal steps         │     │
│  │    - Quality: Output quality score                  │     │
│  └────────────────────────────────────────────────────┘     │
└────────────────────────┬─────────────────────────────────────┘
                         │ 5. Reward + Trace
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  Experience Storage (Memento)                │
│  ┌────────────────────────────────────────────────────┐     │
│  │  Store Case:                                        │     │
│  │    - task_id, query, retrieved_cases                │     │
│  │    - plan_json, meta_trace, executor_trace          │     │
│  │    - tool_history, result, reward                   │     │
│  │    - label: "positive" (R > threshold) or "negative"│     │
│  │    - metadata: tags, tools, execution_time          │     │
│  └────────────────────────────────────────────────────┘     │
└────────────────────────┬─────────────────────────────────────┘
                         │ 6. Training Data
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                  Retriever Training (Periodic)               │
│  ┌────────────────────────────────────────────────────┐     │
│  │  1. Collect training batch (e.g., 100 cases)        │     │
│  │  2. Create positive/negative pairs                  │     │
│  │  3. Train neural retriever                          │     │
│  │  4. Validate on held-out set                        │     │
│  │  5. Deploy updated retriever                        │     │
│  └────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

1. **Tool Integration**: Use adapter pattern to connect Memento's MCP tools with ARE's native tools
2. **Memory Scope**: Store scenario-level experiences (not individual tool calls)
3. **Retrieval Strategy**: Query-driven retrieval before planning (not during execution)
4. **Training Frequency**: Batch training every N scenarios (e.g., N=50-100)
5. **Reward Function**: Start with binary, evolve to shaped rewards

---

## Data Flow

### Phase 1: Execution with Memory

```python
# Detailed execution flow

1. User submits task to ARE environment
   Input: "Find all emails from alice@example.com in the last week"
   
2. ARE loads scenario and initializes apps
   Available tools: [email_search, email_read, time_now, ...]
   
3. Memory retrieval layer activates
   a. Encode task: embed("Find emails from alice@example.com in last week")
   b. Query memory store: retrieve_similar(embedding, k=8)
   c. Return cases: [
        Case(query="Find emails from bob@...", score=0.89),
        Case(query="Search for messages from...", score=0.85),
        ...
      ]
   
4. Hierarchical agent receives:
   - Task description
   - Retrieved cases (with plans and outcomes)
   - Available tools from ARE
   
5. Meta-Planner generates plan:
   prompt = f"""
   Task: {task}
   
   Similar Past Tasks:
   {format_cases(retrieved_cases)}
   
   Available Tools:
   {format_tools(are_tools)}
   
   Generate a step-by-step plan.
   """
   
   plan = planner.generate(prompt)
   # Output: [
   #   {"step": 1, "action": "get_current_time", "args": {}},
   #   {"step": 2, "action": "calculate_time_range", "args": {"days": 7}},
   #   {"step": 3, "action": "email_search", "args": {"sender": "alice@...", "after": "..."}}
   # ]
   
6. Executor runs plan:
   for step in plan:
       # Call ARE tool through adapter
       result = are_tool_adapter.call(step.action, step.args)
       
       # Update context
       context.append(result)
       
       # Handle notifications from ARE
       if notifications := notification_system.get_pending():
           context.append(notifications)
   
7. ARE validates result:
   is_correct = scenario.validate(executor.final_output)
   
8. Compute reward:
   reward = compute_reward(
       success=is_correct,
       num_steps=len(plan),
       execution_time=elapsed_time,
       optional_steps=scenario.optimal_steps
   )
   
9. Store experience:
   case = {
       "task_id": scenario.scenario_id,
       "query": task,
       "retrieved_cases": retrieved_cases,
       "plan": plan,
       "trace": executor.trace,
       "result": executor.final_output,
       "reward": reward,
       "label": "positive" if reward > 0.5 else "negative",
       "metadata": {
           "scenario_id": scenario.scenario_id,
           "tags": scenario.tags,
           "tools_used": extract_tools(executor.trace),
           "execution_time": elapsed_time,
           "timestamp": now()
       }
   }
   
   memory_store.add(case)
```

### Phase 2: Training

```python
# Periodic retriever training

1. Accumulate training batch:
   if len(memory_store.untrained_cases) >= TRAINING_BATCH_SIZE:
       trigger_training()

2. Prepare training data:
   positive_pairs = []
   negative_pairs = []
   
   for case in training_batch:
       query = case.query
       
       if case.label == "positive":
           # This case is a good match for this query
           positive_pairs.append((query, case))
           
           # Also create hard negatives: similar queries but different solutions
           for other_case in find_similar_queries(query, exclude=case):
               if other_case.label == "negative":
                   negative_pairs.append((query, other_case))
       
       else:
           # This case is NOT a good match
           negative_pairs.append((query, case))

3. Train retriever:
   for epoch in range(NUM_EPOCHS):
       for batch in dataloader:
           # Forward pass
           logits = retriever.forward(batch.query_embeddings, batch.case_embeddings)
           
           # Compute loss
           loss = cross_entropy(logits, batch.labels)
           
           # Backward pass
           loss.backward()
           optimizer.step()
   
4. Validate:
   val_metrics = evaluate_retriever(retriever, validation_set)
   
   if val_metrics.accuracy > THRESHOLD:
       save_checkpoint(retriever, f"retriever_v{version}.pt")
       deploy_to_production(retriever)
```

---

## Implementation Components

### 1. ARE-Memento Bridge

Adapter to connect ARE's tools with Memento's agent.

```python
from typing import List, Dict, Any
from are.simulation.tools import Tool
from Memento.client.mcp_toolkit import MCPToolkit

class AREMementoAdapter:
    """
    Adapter between ARE tools and Memento's MCP protocol.
    """
    
    def __init__(self, are_tools: List[Tool]):
        self.are_tools = {tool.name: tool for tool in are_tools}
        self.mcp_toolkit = MCPToolkit()
        
        # Register ARE tools in MCP format
        for tool in are_tools:
            self.mcp_toolkit.register_tool(
                name=tool.name,
                description=tool.description,
                parameters=self._convert_params(tool.parameters),
                handler=lambda args, tool=tool: self._execute_are_tool(tool, args)
            )
    
    def _convert_params(self, are_params) -> Dict[str, Any]:
        """Convert ARE parameter format to MCP format."""
        mcp_params = {
            "type": "object",
            "properties": {},
            "required": []
        }
        
        for param in are_params:
            mcp_params["properties"][param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.required:
                mcp_params["required"].append(param.name)
        
        return mcp_params
    
    def _execute_are_tool(self, tool: Tool, args: Dict[str, Any]) -> Any:
        """Execute ARE tool and return result."""
        return tool.execute(**args)
    
    def get_mcp_toolkit(self) -> MCPToolkit:
        """Get MCP toolkit for Memento agent."""
        return self.mcp_toolkit
```

### 2. Memory-Augmented ARE Agent

```python
from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent
from are.simulation.agents.agent_execution_result import AgentExecutionResult
from Memento.client.parametric_memory_cbr import HierarchicalClientWithMemory
from Memento.memory.parametric_memory import CaseRetriever

class MemoryAugmentedAREAgent(RunnableARESimulationAgent):
    """
    ARE agent with Memento-style memory.
    """
    
    def __init__(
        self,
        memory_retriever: CaseRetriever,
        memory_store: "MemoryStore",
        planner_model: str = "gpt-4-turbo",
        executor_model: str = "o3-mini",
        memory_k: int = 8
    ):
        self.memory_retriever = memory_retriever
        self.memory_store = memory_store
        self.memory_k = memory_k
        
        # Memento hierarchical agent
        self.agent = HierarchicalClientWithMemory(
            planner_model=planner_model,
            executor_model=executor_model,
            memory_retriever=memory_retriever,
            memory_top_k=memory_k
        )
    
    def run_scenario(
        self,
        scenario: "Scenario",
        notification_system: "BaseNotificationSystem | None",
        initial_agent_logs: list | None = None,
    ) -> AgentExecutionResult:
        """
        Run scenario with memory augmentation.
        """
        start_time = time.time()
        
        # 1. Convert ARE tools to MCP format
        are_tools = scenario.get_tools()
        adapter = AREMementoAdapter(are_tools)
        self.agent.set_mcp_toolkit(adapter.get_mcp_toolkit())
        
        # 2. Retrieve relevant memories
        task_query = self._create_query(scenario)
        retrieved_cases = self.memory_retriever.retrieve(
            query=task_query,
            case_pool=self.memory_store.get_all_cases(),
            k=self.memory_k
        )
        
        # 3. Run Memento agent with ARE tools
        result = self.agent.run(
            query=task_query,
            retrieved_cases=retrieved_cases
        )
        
        # 4. Validate with ARE
        is_correct = scenario.validate(result.answer)
        
        # 5. Compute reward
        reward = self._compute_reward(
            success=is_correct,
            num_steps=len(result.trace),
            execution_time=time.time() - start_time,
            scenario=scenario
        )
        
        # 6. Store experience
        self._store_experience(
            scenario=scenario,
            query=task_query,
            retrieved_cases=retrieved_cases,
            result=result,
            reward=reward,
            success=is_correct
        )
        
        # 7. Return ARE result format
        return AgentExecutionResult(
            success=is_correct,
            final_response=result.answer,
            num_steps=len(result.trace),
            execution_time=time.time() - start_time,
            agent_logs=result.trace
        )
    
    def _create_query(self, scenario: "Scenario") -> str:
        """Create query from scenario."""
        # Extract relevant info from scenario
        return f"Scenario: {scenario.scenario_id}"
    
    def _compute_reward(self, success, num_steps, execution_time, scenario) -> float:
        """Compute shaped reward."""
        # Binary reward
        if not success:
            return 0.0
        
        # Shaped reward components
        success_reward = 1.0
        
        # Efficiency: fewer steps is better
        optimal_steps = getattr(scenario, 'optimal_steps', num_steps)
        efficiency_reward = min(1.0, optimal_steps / num_steps) if optimal_steps > 0 else 0.5
        
        # Speed: faster is better (cap at 60s)
        speed_reward = max(0.0, 1.0 - execution_time / 60.0)
        
        # Weighted combination
        reward = 0.6 * success_reward + 0.25 * efficiency_reward + 0.15 * speed_reward
        
        return reward
    
    def _store_experience(self, scenario, query, retrieved_cases, result, reward, success):
        """Store experience in memory."""
        case = {
            "task_id": scenario.scenario_id,
            "query": query,
            "retrieved_cases": [c.to_dict() for c in retrieved_cases],
            "plan": result.plan if hasattr(result, 'plan') else [],
            "trace": result.trace,
            "result": result.answer,
            "reward": reward,
            "label": "positive" if reward > 0.5 else "negative",
            "metadata": {
                "scenario_id": scenario.scenario_id,
                "tags": [tag.value for tag in scenario.tags] if scenario.tags else [],
                "tools_used": self._extract_tools(result.trace),
                "execution_time": result.execution_time if hasattr(result, 'execution_time') else 0,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        self.memory_store.add(case)
    
    def _extract_tools(self, trace) -> List[str]:
        """Extract tool names from execution trace."""
        tools = set()
        for step in trace:
            if isinstance(step, dict) and 'tool' in step:
                tools.add(step['tool'])
        return list(tools)
```

### 3. Memory Store

```python
import json
from pathlib import Path
from typing import List, Dict, Any

class MemoryStore:
    """
    Persistent storage for agent experiences.
    """
    
    def __init__(self, storage_path: str = "memory_store.jsonl"):
        self.storage_path = Path(storage_path)
        self.cases = []
        self.untrained_cases = []
        
        # Load existing cases
        if self.storage_path.exists():
            self._load()
    
    def add(self, case: Dict[str, Any]):
        """Add a new case to memory."""
        self.cases.append(case)
        self.untrained_cases.append(case)
        
        # Persist to disk
        with open(self.storage_path, 'a') as f:
            f.write(json.dumps(case) + '\n')
    
    def get_all_cases(self) -> List[Dict[str, Any]]:
        """Get all stored cases."""
        return self.cases
    
    def get_untrained_cases(self) -> List[Dict[str, Any]]:
        """Get cases not yet used for training."""
        return self.untrained_cases
    
    def mark_as_trained(self, cases: List[Dict[str, Any]]):
        """Mark cases as used for training."""
        case_ids = {c['task_id'] for c in cases}
        self.untrained_cases = [
            c for c in self.untrained_cases 
            if c['task_id'] not in case_ids
        ]
    
    def _load(self):
        """Load cases from disk."""
        with open(self.storage_path, 'r') as f:
            for line in f:
                self.cases.append(json.loads(line))
    
    def query(
        self, 
        tags: List[str] = None,
        min_reward: float = None,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """Query cases with filters."""
        results = self.cases
        
        if tags:
            results = [
                c for c in results 
                if any(tag in c['metadata']['tags'] for tag in tags)
            ]
        
        if min_reward is not None:
            results = [c for c in results if c['reward'] >= min_reward]
        
        if limit:
            results = results[:limit]
        
        return results
```

### 4. Training Pipeline

```python
from Memento.memory.parametric_memory import CaseRetriever
from torch.utils.data import Dataset, DataLoader
import torch

class CaseRetrievalDataset(Dataset):
    """Dataset for training case retriever."""
    
    def __init__(self, cases: List[Dict[str, Any]], is_positive: bool):
        self.cases = cases
        self.is_positive = is_positive
    
    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        case = self.cases[idx]
        
        # Format as retrieval training example
        query_text = case['query']
        case_text = self._format_case(case)
        label = 1 if self.is_positive else 0
        
        return {
            'query': query_text,
            'case': case_text,
            'label': label
        }
    
    def _format_case(self, case):
        """Format case for retrieval."""
        return f"[CASE] {case['query']} [PLAN] {json.dumps(case['plan'])}"

class RetrieverTrainer:
    """Trainer for parametric memory retriever."""
    
    def __init__(
        self,
        memory_store: MemoryStore,
        retriever: CaseRetriever,
        batch_size: int = 16,
        learning_rate: float = 2e-5
    ):
        self.memory_store = memory_store
        self.retriever = retriever
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.optimizer = torch.optim.AdamW(
            retriever.model.parameters(),
            lr=learning_rate
        )
    
    def train_iteration(self):
        """Single training iteration."""
        # Get untrained cases
        untrained = self.memory_store.get_untrained_cases()
        
        if len(untrained) < self.batch_size:
            return  # Not enough data yet
        
        # Prepare training data
        positive_cases = [c for c in untrained if c['label'] == 'positive']
        negative_cases = [c for c in untrained if c['label'] == 'negative']
        
        # Balance classes
        min_size = min(len(positive_cases), len(negative_cases))
        positive_cases = positive_cases[:min_size]
        negative_cases = negative_cases[:min_size]
        
        # Create datasets
        pos_dataset = CaseRetrievalDataset(positive_cases, is_positive=True)
        neg_dataset = CaseRetrievalDataset(negative_cases, is_positive=False)
        
        # Train
        for epoch in range(5):  # Quick fine-tuning
            for batch in DataLoader(pos_dataset, batch_size=self.batch_size):
                loss = self._train_batch(batch)
            
            for batch in DataLoader(neg_dataset, batch_size=self.batch_size):
                loss = self._train_batch(batch)
        
        # Mark as trained
        self.memory_store.mark_as_trained(untrained)
        
        # Save checkpoint
        self.retriever.save(f"retriever_checkpoint_{len(self.memory_store.cases)}.pt")
    
    def _train_batch(self, batch):
        """Train on a single batch."""
        self.optimizer.zero_grad()
        
        # Forward pass (implement based on retriever architecture)
        logits = self.retriever.model(batch['query'], batch['case'])
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(logits, batch['label'])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
```

---

## Reward Derivation Strategy

### Binary Reward (Baseline)

```python
def compute_binary_reward(success: bool) -> float:
    """
    Simple binary reward.
    
    Args:
        success: Whether task completed successfully
    
    Returns:
        1.0 if success, 0.0 otherwise
    """
    return 1.0 if success else 0.0
```

**Advantages:**
- ✅ Simple and clear
- ✅ Aligns with ARE validation
- ✅ Easy to interpret

**Disadvantages:**
- ❌ No partial credit
- ❌ Doesn't encourage efficiency
- ❌ All failures equal

### Shaped Reward (Advanced)

```python
def compute_shaped_reward(
    success: bool,
    num_steps: int,
    optimal_steps: int,
    execution_time: float,
    output_quality: float = 1.0
) -> float:
    """
    Compute shaped reward with multiple components.
    
    Args:
        success: Task completion (0 or 1)
        num_steps: Steps taken by agent
        optimal_steps: Optimal number of steps (from oracle)
        execution_time: Time taken (seconds)
        output_quality: Quality score (0-1) if applicable
    
    Returns:
        Reward in [0, 1]
    """
    if not success:
        # Small reward for partial progress
        progress_reward = 0.1 * min(1.0, num_steps / optimal_steps)
        return progress_reward
    
    # Success components
    success_reward = 0.5  # Base reward for completion
    
    # Efficiency: reward using fewer steps
    efficiency_reward = 0.25 * (optimal_steps / max(num_steps, 1))
    
    # Speed: reward faster execution (normalize by 60s)
    speed_reward = 0.15 * max(0.0, 1.0 - execution_time / 60.0)
    
    # Quality: reward better outputs
    quality_reward = 0.10 * output_quality
    
    total_reward = success_reward + efficiency_reward + speed_reward + quality_reward
    
    return min(1.0, total_reward)
```

**Advantages:**
- ✅ Encourages efficiency
- ✅ Rewards speed
- ✅ Partial credit for attempts
- ✅ Richer learning signal

**Disadvantages:**
- ❌ More complex to tune
- ❌ Requires oracle information
- ❌ May over-optimize for speed vs correctness

### Recommended Approach

**Phase 1** (Weeks 1-2): Binary reward
- Focus on getting basic integration working
- Ensure data collection pipeline is solid
- Validate retrieval improves over time

**Phase 2** (Weeks 3-4): Add efficiency component
- Incorporate optimal steps from oracle
- Penalize overly long action sequences
- Measure impact on retrieval quality

**Phase 3** (Weeks 5+): Full shaped reward
- Add speed and quality components
- Tune weights based on validation performance
- A/B test different reward functions

---

## Training Pipeline

### Training Schedule

```python
class ContinualLearningScheduler:
    """
    Manages when to trigger retriever training.
    """
    
    def __init__(
        self,
        initial_batch_size: int = 100,
        training_frequency: int = 50,  # Train every N new cases
        validation_frequency: int = 10  # Validate every N training iterations
    ):
        self.initial_batch_size = initial_batch_size
        self.training_frequency = training_frequency
        self.validation_frequency = validation_frequency
        
        self.cases_since_training = 0
        self.total_training_iterations = 0
    
    def should_train(self, num_new_cases: int) -> bool:
        """Check if we should trigger training."""
        self.cases_since_training += num_new_cases
        
        # Wait for initial batch
        if self.total_training_iterations == 0:
            return self.cases_since_training >= self.initial_batch_size
        
        # Subsequent training
        return self.cases_since_training >= self.training_frequency
    
    def on_training_complete(self):
        """Called after training completes."""
        self.cases_since_training = 0
        self.total_training_iterations += 1
    
    def should_validate(self) -> bool:
        """Check if we should run validation."""
        return self.total_training_iterations % self.validation_frequency == 0
```

### Full Training Loop

```python
def run_continual_learning(
    agent: MemoryAugmentedAREAgent,
    scenarios: List["Scenario"],
    scheduler: ContinualLearningScheduler,
    trainer: RetrieverTrainer
):
    """
    Main continual learning loop.
    """
    for i, scenario in enumerate(scenarios):
        print(f"\n{'='*80}")
        print(f"Scenario {i+1}/{len(scenarios)}: {scenario.scenario_id}")
        print(f"{'='*80}\n")
        
        # 1. Run scenario
        result = agent.run_scenario(
            scenario=scenario,
            notification_system=None
        )
        
        print(f"Result: {'✓ Success' if result.success else '✗ Failure'}")
        print(f"Steps: {result.num_steps}")
        print(f"Time: {result.execution_time:.2f}s")
        
        # 2. Check if we should train
        if scheduler.should_train(num_new_cases=1):
            print(f"\n{'='*80}")
            print(f"TRAINING RETRIEVER (iteration {scheduler.total_training_iterations + 1})")
            print(f"{'='*80}\n")
            
            # Train retriever
            trainer.train_iteration()
            
            # Update scheduler
            scheduler.on_training_complete()
            
            # Validate if needed
            if scheduler.should_validate():
                metrics = validate_retriever(agent, validation_scenarios)
                print(f"\nValidation Metrics:")
                print(f"  Accuracy: {metrics['accuracy']:.2%}")
                print(f"  Precision@8: {metrics['precision@8']:.2%}")
                print(f"  Recall@8: {metrics['recall@8']:.2%}")
        
        print(f"\nTotal cases in memory: {len(agent.memory_store.cases)}")
        print(f"Cases since last training: {scheduler.cases_since_training}\n")
```

---

## Deployment Architecture

### Production Setup

```
┌─────────────────────────────────────────────────────────┐
│                     Load Balancer                        │
└───────────────────────┬─────────────────────────────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
         ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Agent 1   │ │   Agent 2   │ │   Agent 3   │
│  (GPU pod)  │ │  (GPU pod)  │ │  (GPU pod)  │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │
       └───────────────┼───────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Shared Memory Store (Redis)                 │
│  - Cases stored with embeddings                          │
│  - Supports concurrent reads/writes                      │
│  - Persistent storage to disk                            │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│          Training Pipeline (Separate Service)            │
│  - Polls for new cases                                   │
│  - Triggers training when threshold reached              │
│  - Deploys updated retriever model                       │
└─────────────────────────────────────────────────────────┘
```

### Infrastructure Requirements

**Compute:**
- Agent execution: 4-8 vCPU, 16-32GB RAM per instance
- GPU for retriever inference: 1x T4 or V100
- Training: 1x A100 (40GB) for batch training

**Storage:**
- Memory store: ~1GB per 10K cases
- Model checkpoints: ~500MB per version
- Execution traces: ~10MB per 1K runs

**Networking:**
- Low latency to LLM APIs (OpenAI, Anthropic)
- Redis cluster for distributed memory

---

## Potential Bottlenecks

### 1. Memory Retrieval Latency

**Problem**: Retrieval adds latency to each scenario execution.

**Impact:**
- K=8 retrieval: +50-200ms per query
- Embedding computation: +20-50ms
- Database query: +30-100ms

**Solutions:**
- ✅ Cache embeddings for common queries
- ✅ Use approximate nearest neighbor search (FAISS, Annoy)
- ✅ Precompute embeddings for case pool
- ✅ Batch retrievals when possible

**Implementation:**
```python
from faiss import IndexFlatL2
import numpy as np

class FastMemoryRetriever:
    """Fast retrieval using FAISS."""
    
    def __init__(self, embedding_dim: int = 768):
        self.index = IndexFlatL2(embedding_dim)
        self.cases = []
        self.embeddings_cache = {}
    
    def add_case(self, case: Dict, embedding: np.ndarray):
        """Add case with precomputed embedding."""
        self.cases.append(case)
        self.index.add(embedding.reshape(1, -1))
    
    def retrieve(self, query_embedding: np.ndarray, k: int = 8):
        """Fast retrieval using FAISS."""
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return [self.cases[i] for i in indices[0]]
```

### 2. Training Data Imbalance

**Problem**: Success rate may be very high/low, causing class imbalance.

**Impact:**
- If 90% success: Model learns to always predict "positive"
- If 10% success: Model learns to always predict "negative"
- Poor generalization to new queries

**Solutions:**
- ✅ Balanced sampling during training
- ✅ Class weights in loss function
- ✅ Data augmentation (paraphrase queries)
- ✅ Hard negative mining

**Implementation:**
```python
def prepare_balanced_batch(cases: List[Dict], batch_size: int = 32):
    """Create balanced training batch."""
    positive = [c for c in cases if c['label'] == 'positive']
    negative = [c for c in cases if c['label'] == 'negative']
    
    # Undersample majority class
    min_size = min(len(positive), len(negative))
    positive = random.sample(positive, min_size)
    negative = random.sample(negative, min_size)
    
    # Combine and shuffle
    balanced = positive + negative
    random.shuffle(balanced)
    
    return balanced[:batch_size]
```

### 3. Context Window Overflow

**Problem**: Retrieved cases + task + tools may exceed context limit.

**Impact:**
- Truncated context = lost information
- Poor plan quality
- Increased errors

**Solutions:**
- ✅ Retrieve fewer cases (K=3-5 instead of 8-10)
- ✅ Summarize retrieved cases (extract key points)
- ✅ Filter cases by relevance threshold
- ✅ Prioritize recent + high-reward cases

**Implementation:**
```python
def smart_case_filtering(
    cases: List[Dict],
    max_tokens: int = 4000,
    tokenizer = None
) -> List[Dict]:
    """Filter cases to fit context window."""
    selected = []
    total_tokens = 0
    
    # Sort by score and recency
    cases = sorted(
        cases,
        key=lambda c: (c.get('score', 0), c['metadata']['timestamp']),
        reverse=True
    )
    
    for case in cases:
        case_tokens = len(tokenizer.encode(str(case)))
        
        if total_tokens + case_tokens > max_tokens:
            break
        
        selected.append(case)
        total_tokens += case_tokens
    
    return selected
```

### 4. Cold Start Problem

**Problem**: No memories initially, so no improvement at start.

**Impact:**
- First 50-100 scenarios get no memory benefit
- Training requires minimum data threshold
- Slow initial learning

**Solutions:**
- ✅ Pre-populate with human-labeled examples
- ✅ Use few-shot prompts until memory accumulates
- ✅ Bootstrap from similar task datasets
- ✅ Lower initial training threshold

**Implementation:**
```python
def bootstrap_memory(
    memory_store: MemoryStore,
    bootstrap_examples: List[Dict]
):
    """Pre-populate memory with seed examples."""
    print(f"Bootstrapping with {len(bootstrap_examples)} examples...")
    
    for example in bootstrap_examples:
        # Convert to memory format
        case = {
            "task_id": f"bootstrap_{example['id']}",
            "query": example['query'],
            "plan": example['plan'],
            "result": example['result'],
            "reward": 1.0,  # Assume successful examples
            "label": "positive",
            "metadata": {
                "source": "bootstrap",
                "tags": example.get('tags', []),
                "tools_used": example.get('tools', [])
            }
        }
        
        memory_store.add(case)
    
    print(f"Memory initialized with {len(memory_store.cases)} cases")
```

### 5. Catastrophic Forgetting

**Problem**: Training on new data may degrade performance on old tasks.

**Impact:**
- Retriever forgets how to handle early task types
- Performance regresses on previously mastered skills
- Non-monotonic learning curve

**Solutions:**
- ✅ Replay buffer: Include old examples in training
- ✅ Regularization: L2 penalty on weight changes
- ✅ Elastic weight consolidation (EWC)
- ✅ Continual learning techniques

**Implementation:**
```python
class ReplayBuffer:
    """Store representative examples to prevent forgetting."""
    
    def __init__(self, size: int = 1000):
        self.size = size
        self.buffer = []
    
    def add(self, case: Dict):
        """Add case to replay buffer."""
        if len(self.buffer) < self.size:
            self.buffer.append(case)
        else:
            # Replace random old example
            idx = random.randint(0, self.size - 1)
            self.buffer[idx] = case
    
    def sample(self, n: int) -> List[Dict]:
        """Sample from replay buffer."""
        return random.sample(self.buffer, min(n, len(self.buffer)))

def train_with_replay(
    trainer: RetrieverTrainer,
    new_cases: List[Dict],
    replay_buffer: ReplayBuffer,
    replay_ratio: float = 0.3
):
    """Train with replay to prevent forgetting."""
    # Mix new and old examples
    replay_size = int(len(new_cases) * replay_ratio)
    replay_cases = replay_buffer.sample(replay_size)
    
    training_cases = new_cases + replay_cases
    random.shuffle(training_cases)
    
    # Train
    trainer.train(training_cases)
    
    # Update replay buffer
    for case in new_cases:
        replay_buffer.add(case)
```

---

## Implementation Roadmap

### Phase 1: Basic Integration (Weeks 1-2)

**Goal**: Get end-to-end pipeline working with binary rewards.

**Tasks:**
1. ✅ Implement AREMementoAdapter for tool conversion
2. ✅ Create MemoryAugmentedAREAgent wrapper
3. ✅ Implement MemoryStore with JSONL persistence
4. ✅ Set up binary reward computation
5. ✅ Test on subset of ARE scenarios (10-20)
6. ✅ Validate data collection pipeline

**Success Criteria:**
- Agent successfully runs ARE scenarios
- Experiences stored in memory
- No crashes or data corruption

**Deliverables:**
- Working prototype code
- Initial memory dataset (50+ cases)
- Technical documentation

### Phase 2: Memory Retrieval (Weeks 3-4)

**Goal**: Integrate parametric memory retrieval.

**Tasks:**
1. ✅ Port Memento's CaseRetriever to project
2. ✅ Implement query embedding pipeline
3. ✅ Add retrieval before planning step
4. ✅ Format retrieved cases for prompts
5. ✅ Benchmark retrieval latency
6. ✅ Optimize with caching/indexing

**Success Criteria:**
- Retrieval completes in <200ms
- Cases relevant to queries (manual inspection)
- Memory context properly formatted

**Deliverables:**
- Retrieval module
- Performance benchmarks
- Example retrieval outputs

### Phase 3: Training Loop (Weeks 5-6)

**Goal**: Implement continual learning with periodic training.

**Tasks:**
1. ✅ Create CaseRetrievalDataset
2. ✅ Implement RetrieverTrainer
3. ✅ Set up training scheduler
4. ✅ Add validation on held-out scenarios
5. ✅ Implement model checkpointing
6. ✅ Test full training loop

**Success Criteria:**
- Retriever trains without errors
- Validation accuracy improves over time
- Models saved/loaded correctly

**Deliverables:**
- Training pipeline code
- Learning curves
- Trained retriever checkpoints

### Phase 4: Shaped Rewards (Weeks 7-8)

**Goal**: Enhance reward function for better learning signal.

**Tasks:**
1. ✅ Implement shaped reward computation
2. ✅ Extract optimal steps from oracle traces
3. ✅ Tune reward component weights
4. ✅ Compare shaped vs binary rewards
5. ✅ Analyze impact on retrieval quality

**Success Criteria:**
- Shaped rewards correlate with task quality
- Improved learning efficiency vs baseline
- Higher validation performance

**Deliverables:**
- Reward analysis report
- A/B test results
- Updated reward function

### Phase 5: Optimization (Weeks 9-10)

**Goal**: Address bottlenecks and improve efficiency.

**Tasks:**
1. ✅ Implement FAISS for fast retrieval
2. ✅ Add embedding caching
3. ✅ Balance training data
4. ✅ Smart case filtering for context
5. ✅ Bootstrap with seed examples
6. ✅ Add replay buffer

**Success Criteria:**
- Retrieval latency <100ms
- Training handles class imbalance
- No catastrophic forgetting observed

**Deliverables:**
- Optimized retrieval module
- Training improvements
- Performance comparison

### Phase 6: Evaluation (Weeks 11-12)

**Goal**: Comprehensive evaluation on GAIA2 benchmark.

**Tasks:**
1. ✅ Run full GAIA2 validation (200 scenarios)
2. ✅ Compare with baseline (no memory)
3. ✅ Analyze learning curves
4. ✅ Error analysis on failure cases
5. ✅ Generate final report

**Success Criteria:**
- Statistically significant improvement over baseline
- Learning curve shows monotonic improvement
- Clear understanding of failure modes

**Deliverables:**
- Evaluation report
- Learning curve plots
- Error analysis document
- Final presentation

---

## Code Examples

### Complete Integration Example

```python
# main.py - Full integration example

from are.simulation.environment import Environment, EnvironmentConfig
from are.simulation.scenarios.registration import get_all_scenarios
from Memento.memory.parametric_memory import CaseRetriever

# 1. Initialize components
memory_store = MemoryStore(storage_path="memory_store.jsonl")
retriever = CaseRetriever.load("models/retriever_initial.pt")

agent = MemoryAugmentedAREAgent(
    memory_retriever=retriever,
    memory_store=memory_store,
    planner_model="gpt-4-turbo",
    executor_model="o3-mini",
    memory_k=8
)

scheduler = ContinualLearningScheduler(
    initial_batch_size=100,
    training_frequency=50
)

trainer = RetrieverTrainer(
    memory_store=memory_store,
    retriever=retriever,
    batch_size=16,
    learning_rate=2e-5
)

# 2. Load scenarios
scenarios = get_all_scenarios()[:500]  # First 500 scenarios

# 3. Run continual learning
run_continual_learning(
    agent=agent,
    scenarios=scenarios,
    scheduler=scheduler,
    trainer=trainer
)

# 4. Final evaluation
validation_scenarios = get_all_scenarios()[500:700]  # Hold-out set
metrics = evaluate(agent, validation_scenarios)

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
print(f"Validation Accuracy: {metrics['accuracy']:.2%}")
print(f"Average Steps: {metrics['avg_steps']:.1f}")
print(f"Average Time: {metrics['avg_time']:.1f}s")
print(f"Total Cases in Memory: {len(memory_store.cases)}")
print(f"Training Iterations: {scheduler.total_training_iterations}")
```

---

## Summary

This architecture enables:

1. ✅ **Dynamic Evaluation**: ARE provides complex, realistic scenarios
2. ✅ **Memory-Based Learning**: Memento stores and retrieves experiences
3. ✅ **Continual Improvement**: Periodic training updates retrieval policy
4. ✅ **No Fine-Tuning**: Base LLMs remain unchanged
5. ✅ **Scalable**: Works across diverse task types

**Expected Benefits:**
- 📈 Improved success rate over time (target: +10-20% over baseline)
- ⚡ Faster task completion as strategies are learned
- 🎯 Better generalization to novel tasks
- 💾 Reusable memory across sessions

**Key Risks:**
- 🔴 Cold start: Initial performance same as baseline
- 🟠 Retrieval latency: May slow down execution
- 🟡 Memory scalability: Storage/retrieval costs grow with data

**Next Steps:**
1. Review this architecture with the team
2. Start Phase 1 implementation (basic integration)
3. Set up evaluation infrastructure
4. Begin data collection
