from collections import deque
import numpy as np
import torch
import torch.nn as nn
from thoughtframe.sensor.interface import AcousticChunkProcessor

class ComplexModReLU(nn.Module):
    """
    The "Circle of Silence" Activation.
    
    Logic:
    - If a complex vector's magnitude is inside the circle (radius 'b'), 
      it is considered noise/destructive interference -> Output 0.
    - If it 'escapes' the circle, it is considered a valid signal/constructive interference.
    - We preserve the PHASE (direction) perfectly, only reducing magnitude.
    """
    def __init__(self, threshold=0.5):
        super().__init__()
        self.radius = nn.Parameter(torch.tensor([threshold]))

    def forward(self, z: torch.Tensor):
        # z shape: [Batch, Channels] (Complex64/128)
        mag = torch.abs(z)
        
        # The gating logic: (Magnitude - Radius), clamped at 0.
        # This is standard ReLU applied to the radial distance.
        active_mag = torch.relu(mag - self.radius)
        
        # Reconstruct the vector: New_Mag * Unit_Vector
        # We add 1e-6 to avoid div/0 errors on silence
        return z * (active_mag / (mag + 1e-6))

class LearnedBeamformerLayer(nn.Module):
    """
    The "Steering" Layer.
    
    Logic:
    - Input: Concatenated raw ears (The Towed Array).
    - Weights: Complex-valued steering vectors.
    - Operation: Dot product.
    
    If Weight_i aligns with the phase delays of the Input, the dot product 
    constructively interferes (large magnitude). If they mismatch, they cancel.
    This layer doesn't just sum; it 'looks' in specific directions.
    """
    def __init__(self, n_inputs, n_beams):
        super().__init__()
        # Complex weights initialized with random phase/magnitude
        # In a real towed array, these would initialize to specific geometric delays
        self.weights = nn.Parameter(torch.randn(n_beams, n_inputs, dtype=torch.cfloat))
        
    def forward(self, x):
        # x: [Batch, n_inputs]
        # output: [Batch, n_beams]
        # We use standard matrix multiplication which sums over the input dimension
        return torch.matmul(x, self.weights.T)

class PhaseBeamformingProcessor(AcousticChunkProcessor):
    """
    ARCHITECTURAL DOCUMENTATION
    ---------------------------
    1. Input Strategy: Towed Array (Concatenation)
       - We do NOT sum ears initially. We preserve the array geometry.
       - 5 Ears = 5 distinct channels of phase info.
       
    2. Layer 1: The "Learned Steerer"
       - Dense Complex Matrix Multiply.
       - Each neuron learns a set of phase delays (a "look direction").
       - effectively a learned Phased Array.
       
    3. Activation: ModReLU ("Escape the Circle")
       - Filters out destructive interference.
       - Only coherent signals survive to the next layer.
       
    4. Layer 2: Overlapping Consensus
       - Sparse, overlapping connections.
       - Validates if adjacent "beams" agree on a structure.
       
    5. Feedback: Explaining Away
       - We maintain a running complex mean of the input field.
       - Subtraction happens in complex space (Interference).
    """

    OP_NAME = "phase_beamformer_v2"

    def __init__(self, cfg, sensor):
        super().__init__()
        self.fs = sensor.fs
        self.chunk_size = sensor.chunk_size
        
        # --- Physical Array Setup ---
        self.n_ears = cfg.get("n_ears", 5) 
        self.n_beams = cfg.get("n_beams", 8)  # Number of "directions" to look in
        
        # --- The Network (PyTorch) ---
        self.device = cfg.get("device", "cpu")
        
        # Layer 1: Steering / Beamforming
        self.steerer = LearnedBeamformerLayer(self.n_ears, self.n_beams)
        self.detector = ComplexModReLU(threshold=cfg.get("silence_radius", 0.5))
        
        # Layer 2: Overlapping Groups (The "Consensus" Layer)
        # Instead of fully connected, we define groups of beams that must agree.
        # e.g., Group 0 listens to Beams [0, 1, 2]
        self.consensus_groups = [
            [0, 1, 2],
            [2, 3, 4],
            [4, 5, 6],
            [5, 6, 7]
        ]
        
        # Move to device
        self.steerer.to(self.device)
        self.detector.to(self.device)
        
        # --- Feedback State (The Ghost Image) ---
        # We track the "Expected Input Field" to spot anomalies (residuals)
        self.expected_field = None 
        self.alpha = cfg.get("adaptation_rate", 0.05) 
        
        # Standard buffering
        self.window_sec = cfg.get("window_sec", 2.0)
        self.n_chunks = int((self.window_sec * self.fs) / self.chunk_size)
        self.buffer = deque(maxlen=self.n_chunks)

    @classmethod
    def from_config(cls, cfg, sensor):
        return cls(cfg, sensor)

    def process(self, chunk: np.ndarray, analysis) -> None:
        # Convert chunk to tensor
        x_real = torch.from_numpy(chunk).float().to(self.device)
        
        # --- 1. Synthesize the "Ears" (Simulation Step) ---
        # In a real system, 'chunk' would already be multi-channel.
        # Here, we simulate 5 ears by adding phase delays to the single chunk.
        # This represents the physical arrival time differences.
        
        # Transform to freq domain to fake phase
        fft = torch.fft.rfft(x_real)
        
        ear_inputs = []
        for i in range(self.n_ears):
            # Apply a fake geometric delay (phase rotation)
            # In reality, this IS your raw data.
            angle = i * (np.pi / 8) 
            phase_shift = torch.exp(1j * torch.tensor(angle))
            ear_inputs.append(fft * phase_shift)
            
        # [Batch=1, Ears=5, Freq_Bins] -> collapse freq for this abstract demo
        # We are simplifying (Freq, Time) -> Single Complex Vector per ear for the net
        # to process "current state".
        current_state = torch.stack([e.mean() for e in ear_inputs]).unsqueeze(0) 
        # Shape: [1, 5] (Complex64)
        
        with torch.no_grad():
            
            # --- 2. Feedback / Interference Check ---
            # "Explaining Away": Subtract expectation from observation.
            # If the world is static, residual -> 0.
            # If input moves/changes phase, residual spikes.
            
            if self.expected_field is None:
                self.expected_field = current_state
                
            residual = current_state - self.expected_field
            
            # Update expectation (slowly integrate the new reality)
            self.expected_field = self.expected_field + self.alpha * residual

            # --- 3. The Forward Pass (Perception) ---
            
            # A. STEERING (The "Lens")
            # The network tries to find coherence in the *residual* (the new stuff).
            # If residual is random noise, beamforming fails (low mag).
            # If residual is a new object, beamforming reinforces it.
            beam_out = self.steerer(residual)
            
            # B. DETECTION (The "Gate")
            # Apply ModReLU. Only strong, coherent signals pass.
            detected_beams = self.detector(beam_out)
            
            # C. CONSENSUS (The Overlapping Layer)
            # Summing the overlapping groups.
            # If beams 0, 1, and 2 all lit up, Group 0 will be massive.
            group_outputs = []
            for indices in self.consensus_groups:
                # Sum the complex vectors in this group
                group_sum = detected_beams[0, indices].sum()
                group_outputs.append(group_sum)
                
            group_outputs = torch.tensor(group_outputs)

        # --- 4. Expose the "Mind" of the Sensor ---
        
        # Metrics that matter:
        
        # 1. Total Residual Energy: How much "unexplained" stuff is there?
        analysis.metadata["res_energy"] = float(torch.abs(residual).sum())
        
        # 2. Max Beam Energy: Did *any* direction cohere strongly?
        # (This implies a specific object location, even if we don't know where)
        analysis.metadata["max_beam_coherence"] = float(torch.abs(beam_out).max())
        
        # 3. Detection Rate: How many beams escaped the circle?
        # If 0: Silence/Noise. If >0: Structure.
        n_active = (torch.abs(detected_beams) > 0).sum()
        analysis.metadata["active_beams"] = int(n_active)
        
        # 4. Consensus Strength: 
        # If groups are strong, we have spatially broad agreement.
        analysis.metadata["max_consensus"] = float(torch.abs(group_outputs).max())

        # Save state for debug plotting
        analysis._raw_ears = current_state.cpu().numpy()
        analysis._beams = beam_out.cpu().numpy()
        analysis._groups = group_outputs.cpu().numpy()