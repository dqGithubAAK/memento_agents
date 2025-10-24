# Custom Agent Implementation Guide for ARE (Agents Research Environments)

## Table of Contents
1. [Understanding ARE Architecture](#understanding-are-architecture)
2. [How LLMs Connect to the Environment](#how-llms-connect-to-the-environment)
3. [Tasks, Environments, and Evaluation](#tasks-environments-and-evaluation)
4. [Running LLMs with ARE](#running-llms-with-are)
5. [Implementing Custom Agents](#implementing-custom-agents)
6. [Advanced: Custom Agents with Memory](#advanced-custom-agents-with-memory)

---

## Understanding ARE Architecture

### Core Components

ARE (Agents Research Environments) is a sophisticated simulation platform with several key components:

#### 1. **Environment** (`environment.py`)
The `Environment` is the central orchestrator that:
- Manages **time** through a `TimeManager` (simulated time, not real-time)
- Maintains an **event queue** for scheduling actions/events
- Logs all **completed events** in an event log
- Runs an **event loop** that processes events and advances time
- Provides **apps** (tools) to agents for interaction

**Key Concepts:**
```python
# Environment has two loop modes:
# 1. Time-based: Fixed time increments (e.g., 1 second per tick)
# 2. Queue-based: Jumps to next event time (useful for oracle mode)

env = Environment(
    config=EnvironmentConfig(
        start_time=0,
        duration=3600,  # 1 hour in seconds
        time_increment_in_seconds=1,
        oracle_mode=False,  # Oracle mode = pre-scripted events
        queue_based_loop=False,  # Use time-based loop for agents
    )
)
```

#### 2. **Apps** (Tools)
Apps are interactive applications that provide APIs for agent interaction:
- **AgentUserInterface**: Send/receive messages to/from user
- **EmailClient**: Read and send emails
- **Calendar**: Manage events and appointments
- **FileSystem**: Read/write files
- **Messaging**: Send messages via different platforms
- **Shopping, RentAFlat, Cab, etc.**: Domain-specific tools

Each app has methods decorated with `@register_event` that automatically:
- Log the action to the event log
- Record inputs, outputs, and exceptions
- Track execution time

#### 3. **Events**
Events are the basic unit of interaction:
- **Event**: A scheduled action (e.g., "send email at time T")
- **CompletedEvent**: A recorded action after execution
- **OracleEvent**: Pre-scripted agent action (for validation/testing)
- **ValidationEvent**: Checks if task conditions are met

Events form a **DAG (Directed Acyclic Graph)** with dependencies:
```
Event1 → Event2 → Event4
      ↘ Event3 ↗
```

#### 4. **Scenarios**
A Scenario encapsulates:
- **Initial state**: How the world looks at start
- **Apps**: Which tools are available
- **Events**: What happens during execution (user messages, env changes)
- **Validation logic**: How success is measured

**Example Scenario Structure:**
```python
class MyScenario(Scenario):
    scenario_id = "my_custom_scenario"
    
    def init_and_populate_apps(self):
        # Initialize apps with data
        self.apps = [
            EmailClient(name="email"),
            Calendar(name="calendar"),
            AgentUserInterface(name="aui")
        ]
    
    def build_events_flow(self):
        # Define the task and events
        # Agent receives a message at T=0
        # Environment events happen at T=300, etc.
```

#### 5. **Agents**
Agents are the LLM-powered decision makers that:
- Receive messages from the environment
- Decide which tools to call
- Execute actions via app APIs
- Iterate until task completion

---

## How LLMs Connect to the Environment

### The Connection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         Environment                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │  App: Email  │    │  App: Calendar│    │  App: Files  │     │
│  └──────────────┘    └──────────────┘    └──────────────┘     │
│         ▲                    ▲                    ▲            │
│         │                    │                    │            │
│         └────────────────────┴────────────────────┘            │
│                              │                                  │
│                    ┌─────────▼─────────┐                       │
│                    │  AgentUserInterface│                       │
│                    │   (Message Broker) │                       │
│                    └─────────┬─────────┘                       │
└──────────────────────────────┼──────────────────────────────────┘
                               │
                    ┌──────────▼─────────┐
                    │    Agent Runner    │
                    │  (ScenarioRunner)  │
                    └──────────┬─────────┘
                               │
                    ┌──────────▼─────────┐
                    │  Your Custom Agent  │
                    │   (LangGraph, etc.) │
                    │                     │
                    │  ┌──────────────┐  │
                    │  │  LLM Engine   │  │
                    │  │ (OpenAI, etc.)│  │
                    │  └──────────────┘  │
                    └────────────────────┘
```

### Detailed Process

1. **Agent Initialization**
   ```python
   # Agent receives:
   # - List of available tools (from apps)
   # - Initial system prompt
   # - Notification system for real-time events
   
   agent.run_scenario(scenario, notification_system)
   ```

2. **Tool/Function Calling Loop**
   ```
   LOOP:
   1. Agent receives message from environment
   2. LLM generates reasoning + tool call decision
   3. Agent executes tool via app API
   4. App method runs, decorated with @register_event
   5. CompletedEvent logged with:
      - Input arguments
      - Output/return value
      - Execution time
      - Any exceptions
   6. Result returned to agent
   7. Agent decides next action or sends final response
   ```

3. **What Gets Logged**
   Every tool call is logged with:
   ```python
   CompletedEvent(
       event_id="agent-EmailClient.send_email-uuid",
       event_type=EventType.AGENT,
       action=Action(
           app=EmailClient,
           function=send_email,
           args={"to": "user@example.com", "subject": "Hello", ...}
       ),
       metadata=EventMetadata(
           return_value="Email sent successfully",
           exception=None,
           completed=True
       ),
       event_time=1234567890.0  # Unix timestamp
   )
   ```

4. **Tool Definitions**
   Tools are automatically generated from app methods:
   ```python
   class EmailClient(App):
       @register_event(time_manager, operation_type=OperationType.WRITE)
       def send_email(self, to: str, subject: str, body: str) -> str:
           # Implementation
           return "Email sent"
   
   # Automatically becomes:
   {
       "name": "EmailClient__send_email",
       "description": "Send an email to a recipient",
       "parameters": {
           "type": "object",
           "properties": {
               "to": {"type": "string", "description": "..."},
               "subject": {"type": "string", "description": "..."},
               "body": {"type": "string", "description": "..."}
           }
       }
   }
   ```

---

## Tasks, Environments, and Evaluation

### Task Structure

Tasks in ARE are defined through **Scenarios** which specify:

1. **Initial Universe State**
   - Pre-populated emails, calendar events, files
   - Contact lists, conversation histories
   - Domain-specific data (properties to rent, products to buy, etc.)

2. **Task Description**
   - Delivered via `AgentUserInterface.send_message_to_agent()`
   - Can be simple: "Forward all emails from Alice to Bob"
   - Or complex: "Schedule a meeting with the person who sent the most emails this week, but avoid conflicts with my existing meetings"

3. **Dynamic Events**
   - Environment changes over time
   - Example: New email arrives at T=300s
   - Agent must adapt to changing conditions

4. **Validation Logic**
   - Checks if task was completed correctly
   - Can be simple: "Did agent send the right email?"
   - Or complex: "Did agent handle all edge cases and conflicts?"

### Dataset Splits

ARE includes multiple evaluation sets:

#### 1. **Development/Validation Sets**
- Located in: `meta-agents-research-environments/gaia2` (Hugging Face dataset)
- Includes **oracle events** (ground truth agent actions)
- Used for:
  - Model development
  - Debugging
  - Understanding expected behavior
  - Validation before test submission

**Oracle Mode Example:**
```python
# Validation scenarios include ground truth actions
OracleEvent(
    event_id="oracle-1",
    action_desc=ActionDescription(
        app="EmailClient",
        function="send_email",
        args=[
            {"name": "to", "value": "alice@example.com"},
            {"name": "subject", "value": "Re: Meeting"}
        ]
    ),
    event_type=EventType.AGENT
)
# This shows what the agent SHOULD do
```

#### 2. **Test Sets**
- Private, held by Meta/Hugging Face
- No oracle events
- Used for final leaderboard evaluation
- Voluntary submission system

### GAIA2 Benchmark Structure

**800 scenarios across 7 capabilities:**

1. **Execution** (160 scenarios)
   - Multi-step operations
   - State changes
   - Example: "Update all contacts aged 24 or younger to be one year older"

2. **Search** (160 scenarios)
   - Information gathering
   - Multi-source combination
   - Example: "Which city do most of my friends live in?"

3. **Adaptability** (160 scenarios)
   - Dynamic environment changes
   - Handling consequences
   - Example: "Meet friend. If she suggests another time, reschedule."

4. **Time** (160 scenarios)
   - Temporal reasoning
   - Scheduling constraints
   - Example: "Send messages. If no response in 3 minutes, order cab."

5. **Ambiguity** (160 scenarios)
   - Recognizing impossible tasks
   - Asking clarifying questions
   - Example: "Schedule event. Ask if conflicts."

6. **Agent2Agent** (160 scenarios, mini subset)
   - Multi-agent collaboration
   - Communication protocols

7. **Noise** (160 scenarios, mini subset)
   - Robustness to failures
   - API instability

### Evaluation Metrics

The framework tracks:
```python
ScenarioValidationResult(
    success=True,  # Task completed correctly
    achieved_milestones=["step1", "step2"],  # Sub-goals
    exception=None,  # Any errors
    execution_time=45.3,  # Seconds
    num_steps=12,  # Agent iterations
    num_tool_calls=24  # Total API calls
)
```

---

## Running LLMs with ARE

### Quick Start: Using Existing Models

#### 1. **Using API-Based LLMs**

```bash
# OpenAI
export OPENAI_API_KEY="your-key"
uvx --from meta-agents-research-environments are-run \
  -s scenario_find_image_file \
  --agent default \
  --model gpt-4 \
  --provider openai

# Anthropic
export ANTHROPIC_API_KEY="your-key"
uvx --from meta-agents-research-environments are-run \
  -s scenario_find_image_file \
  --agent default \
  --model claude-3-5-sonnet-20241022 \
  --provider anthropic

# Llama API
export LLAMA_API_KEY="your-key"
uvx --from meta-agents-research-environments are-run \
  -s scenario_find_image_file \
  --agent default \
  --model Llama-3.3-70B-Instruct \
  --provider llama-api
```

#### 2. **Using Local Models**

```bash
# Start your local model server (e.g., vLLM, Ollama, LM Studio)
# Example with vLLM:
# vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# Run ARE with local model
uvx --from meta-agents-research-environments are-run \
  -s scenario_find_image_file \
  --agent default \
  --model local-model \
  --provider local \
  --endpoint "http://localhost:8000"
```

#### 3. **Using Hugging Face Inference**

```bash
# Login first
huggingface-cli login

# Use Hugging Face hosted models
uvx --from meta-agents-research-environments are-run \
  -s scenario_find_image_file \
  --agent default \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --provider hyperbolic
```

### Running Benchmarks

```bash
# Run on GAIA2 mini validation set (160 scenarios)
uvx --from meta-agents-research-environments are-benchmark run \
  --hf-dataset meta-agents-research-environments/gaia2 \
  --hf-split validation \
  --hf-config mini \
  --model gpt-4 \
  --provider openai \
  --output_dir ./results

# Run specific capability
uvx --from meta-agents-research-environments are-benchmark run \
  --hf-dataset meta-agents-research-environments/gaia2 \
  --hf-split validation \
  --hf-config execution \
  --model gpt-4 \
  --provider openai \
  --limit 10
```

### Using the GUI

```bash
# Start GUI for interactive exploration
uvx --from meta-agents-research-environments are-gui \
  -s scenario_find_image_file \
  --model gpt-4 \
  --provider openai

# Load GAIA2 scenario from Hugging Face
uvx --from meta-agents-research-environments are-gui \
  -s hf://datasets/meta-agents-research-environments/gaia2/execution/validation/scenario_universe_0_abc123 \
  --ui_view scenarios
```

---

## Implementing Custom Agents

### Approach 1: Extending the Default Agent

The default agent is a ReAct-style agent. You can customize it by:

```python
from are.simulation.agents.default_agent.react_agent import ReactAgent
from are.simulation.agents.are_simulation_agent_config import ARESimulationReactAgentConfig

class MyCustomReactAgent(ReactAgent):
    def __init__(self, config: ARESimulationReactAgentConfig):
        super().__init__(config)
        # Add custom initialization
    
    def custom_system_prompt(self) -> str:
        """Override to customize agent behavior"""
        base_prompt = super().custom_system_prompt()
        return base_prompt + "\nAlways verify results before responding."
```

### Approach 2: Implementing RunnableARESimulationAgent

Create a completely custom agent:

```python
from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent
from are.simulation.agents.agent_execution_result import AgentExecutionResult
from are.simulation.scenarios.scenario import Scenario
from are.simulation.notification_system import BaseNotificationSystem

class MyLangGraphAgent(RunnableARESimulationAgent):
    def __init__(self, llm_config: dict):
        self.llm_config = llm_config
        # Initialize your LangGraph agent here
    
    def run_scenario(
        self,
        scenario: Scenario,
        notification_system: BaseNotificationSystem | None,
        initial_agent_logs: list | None = None,
    ) -> AgentExecutionResult:
        """
        Main entry point for running agent on a scenario.
        
        Args:
            scenario: Contains tasks, tools, and environment
            notification_system: For real-time event notifications
            initial_agent_logs: Previous execution logs (for replay)
        
        Returns:
            AgentExecutionResult with execution details
        """
        # 1. Get tools from scenario
        tools = scenario.get_tools()
        
        # 2. Get initial message/task
        # (This is sent via environment events)
        
        # 3. Run your agent loop
        result = self._run_agent_loop(tools, notification_system)
        
        # 4. Return execution result
        return AgentExecutionResult(
            success=True,
            final_response=result,
            num_steps=10,
            execution_time=45.0
        )
    
    def _run_agent_loop(self, tools, notification_system):
        # Your agent implementation here
        pass
```

### Approach 3: Using External Agent Frameworks (LangGraph, CrewAI, etc.)

**LangGraph Example:**

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from are.simulation.agents.are_simulation_agent import RunnableARESimulationAgent

class LangGraphAREAgent(RunnableARESimulationAgent):
    def __init__(self, model_name: str = "gpt-4"):
        self.model = ChatOpenAI(model=model_name)
        self.agent = None
    
    def run_scenario(self, scenario, notification_system, initial_agent_logs=None):
        # Convert ARE tools to LangChain tools
        tools = self._convert_are_tools_to_langchain(scenario.get_tools())
        
        # Create LangGraph agent
        self.agent = create_react_agent(self.model, tools)
        
        # Subscribe to notifications for dynamic events
        if notification_system:
            notification_system.subscribe(self._handle_notification)
        
        # Run agent loop
        # ... implementation ...
    
    def _convert_are_tools_to_langchain(self, are_tools):
        """Convert ARE tools to LangChain format"""
        from langchain.tools import Tool
        
        langchain_tools = []
        for tool in are_tools:
            langchain_tools.append(Tool(
                name=tool.name,
                func=tool.function,
                description=tool.function_description
            ))
        return langchain_tools
```

### Tool Access Pattern

All agent implementations interact with tools the same way:

```python
# Get tools from scenario
tools = scenario.get_tools()

# Each tool has:
# - name: Unique identifier (e.g., "EmailClient__send_email")
# - function: Callable Python function
# - function_description: What the tool does
# - parameters_schema: JSON schema for arguments

# Example: Calling a tool
for tool in tools:
    if tool.name == "EmailClient__send_email":
        result = tool.function(
            to="user@example.com",
            subject="Hello",
            body="This is a test"
        )
        # Tool call is automatically logged to environment
```

---

## Advanced: Custom Agents with Memory

### Memory Architecture

For agents with self-improvement/memory:

```python
class MemoryAugmentedAgent(RunnableARESimulationAgent):
    def __init__(self, llm_config: dict, memory_config: dict):
        self.llm_config = llm_config
        self.memory_config = memory_config
        
        # Initialize memory components
        self.episodic_memory = []  # Past scenarios
        self.semantic_memory = {}   # Learned strategies
        self.working_memory = {}    # Current task context
    
    def run_scenario(self, scenario, notification_system, initial_agent_logs=None):
        # 1. Retrieve relevant memories
        relevant_memories = self._retrieve_similar_scenarios(scenario)
        
        # 2. Augment prompt with memories
        augmented_prompt = self._create_memory_augmented_prompt(
            scenario, 
            relevant_memories
        )
        
        # 3. Run agent with memory context
        result = self._run_with_memory(
            scenario, 
            augmented_prompt,
            notification_system
        )
        
        # 4. Store execution trace for future learning
        self._store_execution_trace(scenario, result)
        
        return result
    
    def _retrieve_similar_scenarios(self, scenario):
        """Retrieve relevant past experiences"""
        # Implement similarity search over episodic_memory
        # Could use embeddings, keywords, etc.
        pass
    
    def _store_execution_trace(self, scenario, result):
        """Store this execution for future reference"""
        self.episodic_memory.append({
            'scenario_id': scenario.scenario_id,
            'tools_used': result.tools_used,
            'success': result.success,
            'strategy': self._extract_strategy(result),
            'execution_trace': result.execution_trace
        })
        
        # Update semantic memory with learned patterns
        if result.success:
            self._update_learned_strategies(scenario, result)
```

### Accessing Environment Logs

Your agent can access all logged events:

```python
def _run_with_memory(self, scenario, augmented_prompt, notification_system):
    # Get environment reference
    from are.simulation.environment import Environment
    env = ...  # Obtained from scenario runner
    
    # Access event log
    past_events = env.event_log.list_view()
    
    # Filter for agent actions
    agent_actions = [
        event for event in past_events 
        if event.event_type == EventType.AGENT
    ]
    
    # Analyze patterns
    for event in agent_actions:
        tool_name = event.action.function.__name__
        args = event.action.args
        result = event.metadata.return_value
        # Learn from this action
```

### Multi-Task Learning

```python
class MetaLearningAgent(MemoryAugmentedAgent):
    def __init__(self, llm_config: dict, memory_config: dict):
        super().__init__(llm_config, memory_config)
        self.task_performance = {}  # Track per-task metrics
    
    def meta_learn_across_tasks(self, scenarios: list):
        """Learn general strategies across multiple scenarios"""
        
        # Group scenarios by capability
        by_capability = self._group_by_capability(scenarios)
        
        # Extract common patterns per capability
        for capability, cap_scenarios in by_capability.items():
            patterns = self._extract_common_patterns(cap_scenarios)
            self.semantic_memory[capability] = patterns
        
        # Learn cross-capability strategies
        cross_patterns = self._extract_cross_capability_patterns(scenarios)
        self.semantic_memory['general'] = cross_patterns
    
    def _extract_common_patterns(self, scenarios):
        """Identify successful patterns in similar scenarios"""
        successful_traces = [
            trace for trace in scenarios
            if trace['success']
        ]
        
        # Analyze tool sequences
        common_sequences = self._find_common_tool_sequences(successful_traces)
        
        # Identify effective strategies
        strategies = self._cluster_strategies(successful_traces)
        
        return {
            'tool_sequences': common_sequences,
            'strategies': strategies
        }
```

### Integration with External Memory Systems

```python
class VectorMemoryAgent(RunnableARESimulationAgent):
    def __init__(self, llm_config: dict, vector_store_config: dict):
        self.llm_config = llm_config
        
        # Initialize vector store (e.g., Chroma, Pinecone, etc.)
        from chromadb import Client
        self.vector_store = Client()
        self.collection = self.vector_store.create_collection("agent_memory")
        
        # Initialize embedding model
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def _store_execution_trace(self, scenario, result):
        """Store execution trace in vector database"""
        
        # Create document from execution
        doc = {
            'scenario_id': scenario.scenario_id,
            'task': scenario.task_description,
            'tools_used': result.tools_used,
            'success': result.success,
            'trace': result.execution_trace
        }
        
        # Generate embedding
        embedding = self.embedder.encode(
            f"{scenario.task_description} {result.execution_trace}"
        )
        
        # Store in vector database
        self.collection.add(
            documents=[str(doc)],
            embeddings=[embedding.tolist()],
            ids=[scenario.scenario_id]
        )
    
    def _retrieve_similar_scenarios(self, scenario):
        """Retrieve similar scenarios from vector store"""
        
        # Embed current task
        query_embedding = self.embedder.encode(scenario.task_description)
        
        # Query vector store
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5
        )
        
        return results
```

---

## Next Steps

1. **Explore Examples**: Check `examples/` directory for sample implementations
2. **Read Tutorials**: See Jupyter notebooks in this repo
3. **Run Validation**: Test on validation set before test submission
4. **Join Community**: Engage with other researchers on Hugging Face

## Resources

- **Documentation**: https://facebookresearch.github.io/meta-agents-research-environments/
- **Paper**: https://ai.meta.com/research/publications/are-scaling-up-agent-environments-and-evaluations/
- **Leaderboard**: https://huggingface.co/spaces/meta-agents-research-environments/leaderboard
- **Dataset**: https://huggingface.co/datasets/meta-agents-research-environments/gaia2
