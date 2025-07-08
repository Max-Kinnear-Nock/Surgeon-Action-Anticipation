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

<h2>Methodology Overview</h2>

<h3>Dataset</h3>
<ul>
  <li><strong>CholecT50:</strong> 50 annotated laparoscopic cholecystectomy videos with action triplets (Instrument, Verb, Target).</li>
  <li><strong>Split:</strong> 70% training, 10% validation, 20% testing using predefined challenge splits.</li>
  <li><strong>Note:</strong> Data from a single hospital limits generalisability.</li>
</ul>

<h3>Data Preprocessing & Augmentation</h3>
<ul>
  <li>Resized frames to 250×250, then cropped to 224×224.</li>
  <li>Augmentations: random rotation (±5°), horizontal flip, ±10% brightness/contrast/saturation, ±5% hue shift.</li>
  <li>Normalised RGB channels using dataset-specific means and std devs.</li>
  <li><strong>Sliding Window:</strong> Sequences sampled to provide temporal context for model input.</li>
</ul>

<h3>Model Architecture</h3>
<ul>
  <li><strong>Feature Extractor:</strong> Pretrained ResNet-50 with task-specific pointwise convolution.</li>
  <li><strong>Temporal Encoder:</strong> Transformer encoder with causal masking.</li>
  <li><strong>Decoder:</strong> GPT-2-style decoder to anticipate future actions.</li>
  <li><strong>Outputs:</strong> Four MLP classification heads for Instrument, Verb, Target, and Action Triplet.</li>
</ul>

<h3>Loss Functions</h3>
<ul>
  <li><strong>Binary Cross-Entropy with Logits:</strong> Used for multi-label classification.</li>
  <li><strong>Multi-Channel Loss:</strong> Combines discriminative and diversity terms.</li>
  <li><strong>Total Loss:</strong> Recognition + Anticipation + 0.005 × Discriminative + 2 × Diversity</li>
</ul>

<h3>Training Details</h3>
<ul>
  <li>Trained for 20 epochs on an NVIDIA A100 GPU using PyTorch.</li>
  <li>1 frame/sec sampling; 13-frame input sequences (last 3 reserved for anticipation).</li>
  <li><strong>Optimizer:</strong> AdamW with adaptive learning rates.</li>
  <li><strong>Learning Rates:</strong> 1e-4 for ResNet-50 backbone, 1e-3 for other components.</li>
</ul>

<h3>Evaluation</h3>
<ul>
  <li>Evaluated on test set using official CholecT50 scripts.</li>
  <li><strong>Metric:</strong> mean Average Precision (mAP) on action triplet and components (I, V, T).</li>
</ul>

<h3>Model Explainability</h3>
<ul>
  <li><strong>Cumulative accuracy plots:</strong> Assess long-horizon performance.</li>
  <li><strong>Qualitative analysis:</strong> Custom visualisers and block graphs show model behaviour over time.</li>
</ul>

<h3>Ablation Studies</h3>
<ul>
  <li>Varied <strong>context length</strong> and <strong>prediction horizon</strong> to measure impact on anticipation accuracy.</li>
  <li>Tested recognition-only vs. anticipation-only vs. joint training to evaluate loss interaction.</li>
  <li>Each ablation ran for 5 epochs for efficient comparison.</li>
</ul>

<h3>Qualitative Analysis</h3>

<p>
  The model’s behaviour is qualitatively assessed through two visual methods:
</p>

<ul>
  <li>
    <strong>Video Playback:</strong> A custom Python program displays the <em>last recognised frame</em> and the <em>next 10 anticipated frames</em>, helping evaluate temporal consistency in predictions.
  </li>
  <li>
    <strong>Block Graphs:</strong> These show predicted and ground truth action triplets across 11 frames (1 recognised + 10 predicted), with triplet predictions aligned vertically per frame.
  </li>
</ul>

![image](https://github.com/user-attachments/assets/04b8b4b2-7a3e-4ca9-9181-323e7e5797bb)


<p>
  <strong>Above Example:</strong> Each frame is read left to right. Above each frame: frame number. Below:
</p>
<ul>
  <li><strong>Center row:</strong> Model's predicted triplet sequences.</li>
  <li><strong>Bottom row:</strong> Ground truth triplet sequences.</li>
</ul>

<p>
  The red vertical line indicates the division between recognition (left) and anticipation (right). Each coloured block represents a distinct triplet (e.g. grey = <code>&lt;null_instrument, null_verb, null_target&gt;</code>).
</p>






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
