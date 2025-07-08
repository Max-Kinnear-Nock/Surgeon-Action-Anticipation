<p>
  This project explores the application of <strong>Deep Learning</strong> to 
  anticipate surgeons’ next actions during <em>laparoscopic procedures</em>, 
  contributing to the emerging field of <strong>intra-operative AI</strong>. 
  While AI has made major advances in <strong>medical diagnostics</strong>, 
  its integration into the <strong>surgical workflow</strong> remains limited. 
  This work aims to bridge that gap.
</p>


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
