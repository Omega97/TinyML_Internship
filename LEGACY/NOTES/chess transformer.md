
## Architettura

- **Input board encoding**
    
    - Type: tensor di piani della scacchiera.
        
    - Input shape: `B x 24 x 8 x 8`.
        
    - Output shape: `B x 24 x 8 x 8`.
        
    - params: **0**
    
- **Flatten / tokenization**
    
    - Type: reshape.
        
    - Input shape: `B x 24 x 8 x 8`.
        
    - Output shape: `B x 64 x 24`.
        
    - params: **0**
    
- **Linear embedding**
    
    - Type: dense projection.
        
    - Input shape: `B x 64 x 24`.
        
    - Output shape: `B x 64 x 64`.
        
    - params: **1,600**
    
- **Positional embedding**
    
    - Type: learned positional encoding.
        
    - Input shape: `B x 64 x 64`.
        
    - Output shape: `B x 64 x 64`.
        
    - params: **4,096**
    
- **Transformer encoder block 1**
    
    - Type: self-attention + FFN + residuals + LayerNorm.
        
    - Input shape: `B x 64 x 64`.
        
    - Output shape: `B x 64 x 64`.
        
    - params: **33,472**
    
- **Transformer encoder block 2**
    
    - Type: self-attention + FFN + residuals + LayerNorm.
        
    - Input shape: `B x 64 x 64`.
        
    - Output shape: `B x 64 x 64`.
        
    - params: **33,472**
    
- **Mean pooling**
    
    - Type: average over token dimension.
        
    - Input shape: `B x 64 x 64`.
        
    - Output shape: `B x 64`.
        
    - params: **0**
    
- **Shared dense layer**
    
    - Type: MLP projection.
        
    - Input shape: `B x 64`.
        
    - Output shape: `B x 64`.
        
    - params: **4,160**
    
- **Policy head**
    
    - Type: linear output layer.
        
    - Input shape: `B x 64`.
        
    - Output shape: `B x 2048`.
        
    - params: **133,120**
    
- **Value head**
    
    - Type: linear output layer + tanh.
        
    - Input shape: `B x 64`.
        
    - Output shape: `B x 1`.
        
    - params: **65**


**Total params**: 209,985
