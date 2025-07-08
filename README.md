 <h1>Predicting Surgeons’ Next Actions with Deep Learning</h1>

  <p>
    This project explores the application of <strong>Deep Learning</strong> to 
    anticipate surgeons’ next actions during <em>laparoscopic procedures</em>, 
    contributing to the emerging field of <strong>intra-operative AI</strong>. 
    While AI has made major advances in <strong>medical diagnostics</strong>, 
    its integration into the <strong>surgical workflow</strong> remains limited. 
    This work aims to bridge that gap.
  </p>

  <h2>Model Architecture</h2>
  <ul>
    <li><strong>ResNet-50</strong> feature extractor</li>
    <li><strong>Transformer encoder</strong> for temporal understanding</li>
    <li><strong>GPT-2-style decoder</strong> for anticipation</li>
  </ul>

</body>
</html>

![image](https://github.com/user-attachments/assets/04e1fac7-0c10-45ce-9205-857467106640)


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
