import torch
from torch_geometric.data import Data



class ReplayBuffer:

    def __init__(self, capacity=5000):

        self.buffer = deque(maxlen=capacity)

    def store(self, state, layer, node, reward, nextstate, done):
        """
        Stores a transition in the buffer.
        """
        self.buffer.append((state, layer, node, reward, nextstate, done))

    def sample(self, batchsize=64):
        """
        Samples a random batch of experiences.
        """
        return random.sample(self.buffer, min(len(self.buffer), batchsize))

    def size(self):
        """
        Returns the current buffer size.
        """
        return len(self.buffer)


class HGQNAgent:

    def __init__(self, H, L, lr=1e-3, gamma=0.99):
    
        self.H = H
        self.L = L

        self.qnet = HGQN(H, L)

        self.targetqnet = HGQN(H, L)
        self.targetqnet.load_state_dict(self.qnet.state_dict())

        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)

        self.gamma = gamma
        self.epsilon = 1.0  # Exploration probability
        self.edecay = 0.995
        self.emin = 0.1

        self.buffer = ReplayBuffer()

    def act(self, x, edges, lid):
        """
        Epsilon-greedy action selection using hierarchical GNN.
        """
        if random.random() < self.epsilon:
            # Choose a random valid layer
            layer = np.random.randint(0, self.L)

            # Choose a valid node from that layer
            nodes = (lid == layer).nonzero(as_tuple=True)[0]
            if len(nodes) == 0:
                return None, None  # No valid action
            node = np.random.choice(nodes)
        
        else:
            lQ, nQ = self.qnet(graph.x, graph.edges, graph.lid)

            # Select best layer
            layer = torch.argmax(lQ).item()

            # Select best node within the chosen layer
            nodes = (lid == layer).nonzero(as_tuple=True)[0]
            if len(nodes) == 0:
                return None, None  # No valid action
            node = nodes[torch.argmax(nQ[nodes]).item()]

        return layer, node
    
    def train(self, batchsize=64):
        """
        Trains the network using a batch from the replay buffer.
        """
        if self.buffer.size() < batchsize:
            # Not enough experiences to sample from
            return

        # Sample batch from replay buffer
        batch = self.buffer.sample(batchsize)

        graphs, layers, nodes, rewards, graphs_, completed = zip(*batch)

        # Convert batch to PyTorch tensors
        layers    = torch.tensor(layers,  dtype=torch.long)
        nodes     = torch.tensor(nodes,   dtype=torch.long)
        rewards   = torch.tensor(rewards, dtype=torch.float)
        completed = torch.tensor(dones,   dtype=torch.float)

        # Process states (extract node features, edge indices, and layer ids)
        features = torch.cat([g.x     for g in graphs])  
        edges    = torch.cat([g.edges for g in graphs], dim=1)
        lids     = torch.cat([g.lid   for g in graphs])

        # Process next states
        features_ = torch.cat([g.x     for g in graphs_])  
        edges_    = torch.cat([g.edges for g in graphs_], dim=1)  
        lids_     = torch.cat([g.lid   for g in graphs_])

        # Compute target Q-values using the target network
        with torch.no_grad():
            lQ_, nQ_ = self.targetqnet(features_, edges_, lids_)
            mlq_ = lQ_.max(dim=1)[0]
            mnq_ = nQ_.max(dim=1)[0]

            tlq = rewards + self.gamma * mlq_ * (1 - completed)
            tnq = rewards + self.gamma * mnq_ * (1 - completed)

        # Compute predicted Q-values
        lQ, nQ = self.qnet(features, edges, lids)
        plq = lQ.gather(1, layers.unsqueeze(1)).squeeze()
        pnq = nQ.gather(1,  nodes.unsqueeze(1)).squeeze()

        # Compute loss (MSE loss)
        loss = F.mse_loss(plq, tlq) + F.mse_loss(pnq, tnq)

        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon for exploration
        self.epsilon = max(self.epsilon * self.edecay, self.emin)

    def interact(self, params):
    
        graph = dosomething(params)  # Get graph representation
        layer, node = self.act(graph)

        if layer is None or node is None:
            raise Exception
        
        graph_, reward, done = bnb(layer, node)

        self.store(graph, layer, node, reward, graph_, done)
        
        agent.train()
        # Should not be here...
        agent.update()

        print(f"Episode {episode}, Epsilon: {self.epsilon:.2f}")


def get_graph_representation(domain):
    """
    Generates a PyTorch Geometric graph representation of the current 
    Branch-and-Bound (B&B) state for Reinforcement Learning.
    
    Args:
        domain: A dictionary containing info of state.
    
    Returns:
        graph (torch_geometric.data.Data): A graph representation with:
            - x: Node features (lower & upper bounds)
            - edge_index: Graph connectivity
            - batch: Graph batch indices
            - lid: Layer IDs
    """
    
    # Extract node features: each node contains [lower_bound, upper_bound]
    x = __getnodebounds(domain)  # Expected shape: [N, 2]
    
    # Define graph connectivity (edges): Adjacency list for relationships
    edge_index = __getedges(domain)  # Expected shape: [2, E] (src, dest)
    
    # Batch index (for batched training, currently all in one batch)
    batch = torch.zeros(x.shape[0], dtype=torch.long)  # Shape: [N]
    
    # Assign nodes to layers (each node belongs to a specific branching layer)
    lid = __getlayerids(domain)  # Expected shape: [N]

    # Construct graph using PyTorch Geometric
    graph = Data(
        x=torch.FloatTensor(x),              # Node features
        edge_index=torch.LongTensor(edge_index),  # Edge connections
        batch=batch,                         # Batch index
        lid=torch.LongTensor(lid)            # Layer IDs
    )

    return graph


def reward(bounds, newbounds, activenodes, newactivenodes, depth, done):
    """
    Custom reward function for branching heuristic.
    """

    u,  l  = bounds
    u_, l_ = newbounds

    # Compute the bound gap before and after branching
    delta  = u  - l
    delta_ = u_ - l_

    # Reward for closing the relaxation gap
    gapreward = (delta) - delta_) / (delta + 1e-6)

    # Reward for reducing the active search space
    activereward = activenodes - newactivenodes

    # Penalty for increasing depth
    depthpenalty = -0.1 * depth

    # Large reward if optimal solution is found
    donereward = 100 if done else 0

    reward = gapreward + activereward + depthpenalty + donereward

    return reward


def __getedges(layer_sizes):
    """
    """
    edges = []
    noffset = 0  # Keeps track of node indices across layers

    for i in range(len(layer_sizes) - 1):

        start = range(noffset, noffset + layer_sizes[i])  # Nodes in current layer
        end   = range(noffset + layer_sizes[i], noffset + layer_sizes[i] + layer_sizes[i + 1])  # Next layer nodes

        # Create edges from each node in the current layer to each node in the next layer
        for src in start:
            for dst in end:
                # You can exclude stable nodes here or... see above.
                edges.append([src, dst])

        node_offset += layer_sizes[i]  # Update node offset

    return torch.tensor(edges, dtype=torch.long).T  # Transpose for correct shape (2, num_edges)