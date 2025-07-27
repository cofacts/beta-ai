# Copyright 2025 The Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Cofacts AI - Multi-agent fact-checking system for Cofacts platform.

This package provides a hierarchical agent system for fact-checking suspicious messages:
- AI Writer (main orchestrator): Composes fact-check replies and coordinates all sub-agents

The AI Writer can delegate tasks to specialized sub-agents:
- AI Investigator: Deep research using Cofacts DB, external fact-check sources
- AI Verifier: Verifies claims against provided URLs and sources  
- AI Proof-readers: Role-play different political perspectives to test replies

Usage:
    from cofacts_ai import agent
    
    response = await agent.run_async(
        "Please fact-check this message: [suspicious message]"
    )
"""

from .agent import ai_writer

# Export ai_writer as 'agent' to maintain consistency with other agents
agent = ai_writer

__all__ = ["agent"]
