# DINOv2 Inference Skill

When building any inference-related code:

- Always use async/await
- Image input must be preprocessed to 518x518, normalized with mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225]
- Triton expects float32 numpy array shape (1,3,518,518)
- Always include confidence score in response
- Handle Triton connection errors gracefully with retry
- Log inference time for every request
