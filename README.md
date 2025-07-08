Anticipating Surgical Actions Using Deep Learning
Master’s Final Year Project – Biomedical Engineering, King's College London

Project Summary
Short paragraph like the one we prepared earlier, clearly stating the aim and impact.

Objectives
Develop a deep learning model to predict surgeons’ next steps during laparoscopic surgery.

Explore architectural improvements and training strategies that push anticipation performance forward.

Lay the groundwork for future intra-operative AI research.

Model Architecture
  ResNet-50 feature extractor
  Transformer encoder (temporal understanding)
  GPT-2-style decoder for anticipation
(Include diagram here or link to docs/model_architecture.png)

| Metric                  | Value     |
| ----------------------- | --------- |
| mAP for action triplets | 29.0      |
| Optimal context length  | 10 frames |
| Prediction horizon      | 4 frames  |

(Add visualizations or performance charts if available.)

📁 Repository Contents
Folder	Description
models/	Model components
training/	Training + evaluation scripts
utils/	Metrics and helper functions


Incorporate global temporal context (e.g., SuPRA, LoViT).

Train on more diverse surgical procedures for better generalisability.

Focus on clinical accuracy before tackling real-time deployment.

MIT / Academic Use Only (adjust based on your preference)
