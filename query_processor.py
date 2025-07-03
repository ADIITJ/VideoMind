import requests
import json
from datetime import datetime

class QueryProcessor:
    def __init__(self, knowledge_graph):
        self.knowledge_graph = knowledge_graph
        self.ollama_url = "http://localhost:11434/api/generate"

    def process_query(self, query):
        """Process a natural language query."""
        # Parse query with Mistral 7B
        parse_prompt = f"Parse this query into JSON with fields: entity, action, object, time_range, counterfactual (true/false). Query: {query}"
        parsed = self._call_ollama(parse_prompt)
        try:
            parsed_json = json.loads(parsed)
        except:
            return "Error parsing query"

        entity = parsed_json.get("entity")
        time_range = parsed_json.get("time_range")
        counterfactual = parsed_json.get("counterfactual", False)

        if counterfactual:
            return self._handle_counterfactual(query, entity, parsed_json.get("action"))
        else:
            return self._handle_query(entity, time_range)

    def _handle_query(self, entity, time_range):
        """Handle standard queries."""
        if time_range:
            try:
                start = float(time_range[0])
                end = float(time_range[1])
                time_range = (start, end)
            except:
                time_range = None
        events = self.knowledge_graph.query_events(entity, time_range)
        if not events:
            return "No relevant events found."
        descriptions = [event["description"] for event in events]
        return ". ".join(descriptions)

    def _handle_counterfactual(self, query, entity, new_action):
        """Handle counterfactual queries."""
        events = self.knowledge_graph.query_events(entity)
        if not events:
            return "No relevant events to modify."
        # Simulate counterfactual on the latest event
        event_id = max([e for e, d in self.knowledge_graph.graph.nodes(data=True) if d.get("type") == "event"], key=lambda x: self.knowledge_graph.graph.nodes[x]["timestamp"])
        new_graph = self.knowledge_graph.counterfactual(event_id, new_action)
        # Describe counterfactual outcome with Mistral 7B
        counterfactual_prompt = f"Given this modified event: {new_action}. Describe what would have happened: {query}"
        return self._call_ollama(counterfactual_prompt)

    def _call_ollama(self, prompt):
        """Call Mistral 7B via Ollama API."""
        try:
            payload = {
                "model": "mistral:7b-instruct-q4_0",
                "prompt": prompt,
                "stream": False
            }
            response = requests.post(self.ollama_url, json=payload)
            response.raise_for_status()
            return json.loads(response.text).get("response", "")
        except Exception as e:
            return f"Error calling Ollama: {str(e)}"