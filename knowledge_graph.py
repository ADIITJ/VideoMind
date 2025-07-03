import networkx as nx
from datetime import datetime

class TemporalKnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_event(self, timestamp, subject, action, object_, description):
        """Add an event to the graph."""
        event_id = f"event_{len(self.graph.nodes)}"
        self.graph.add_node(event_id, type="event", timestamp=timestamp, description=description)
        self.graph.add_node(subject, type="entity")
        self.graph.add_node(object_, type="entity")
        self.graph.add_edge(subject, event_id, action=action)
        self.graph.add_edge(event_id, object_, action="affects")

    def query_events(self, entity=None, time_range=None):
        """Query events involving an entity or within a time range."""
        results = []
        for node, data in self.graph.nodes(data=True):
            if data.get("type") == "event":
                if (entity is None or entity in self.graph.predecessors(node) or entity in self.graph.successors(node)) and \
                   (time_range is None or (time_range[0] <= data["timestamp"] <= time_range[1])):
                    results.append(data)
        return results

    def counterfactual(self, event_id, new_action=None):
        """Simulate a counterfactual by modifying an event."""
        if event_id not in self.graph.nodes:
            return None
        # Create a copy of the graph
        new_graph = self.graph.copy()
        if new_action:
            # Update event action (simplified)
            for pred, succ, data in new_graph.edges(data=True, nbunch=[event_id]):
                if succ == event_id:
                    data["action"] = new_action
        return new_graph