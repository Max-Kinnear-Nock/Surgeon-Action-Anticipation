**TL;DR**  
> I built an AI model that can predict what a surgeon will do next during keyhole surgery using deep learning.  
> It analyses real surgical videos to anticipate tool use, actions, and anatomy being operated on, which could help improve surgical training or assist during operations.

 
 
 <h1>Predicting Surgeons’ Next Actions with Deep Learning</h1>

  <p>
    This project explores the application of <strong>Deep Learning</strong> to 
    anticipate surgeons’ next actions during <em>laparoscopic procedures</em>, 
    contributing to the emerging field of <strong>intra-operative AI</strong>. 
    While AI has made major advances in <strong>medical diagnostics</strong>, 
    its integration into the <strong>surgical workflow</strong> remains limited. 
    This work aims to bridge that gap.
  </p>

 <h2>CholecT50 Dataset</h2>
<p>
  The <strong>CholecT50</strong> dataset supports deep learning models for 
  <em>fine-grained surgical action recognition</em>. It includes 
  <strong>50 full-length laparoscopic cholecystectomy (LC) surgeries</strong>, 
  each recorded from the surgeon’s perspective. Every video frame is annotated with:
</p>
<ul>
  <li><strong>Surgical phase</strong></li>
  <li><strong>Tool positions</strong></li>
  <li><strong>Action triplets</strong></li>
</ul>
<p>
  The dataset was used in the <strong>CholecT50 Challenge (2021–2022)</strong> to promote 
  the development of high-accuracy intra-operative AI systems. Although all procedures come 
  from the same hospital (limiting generalisability), CholecT50 remains one of the most detailed 
  open benchmarks in <em>Surgical Data Science (SDS)</em>.
  can be accsed here: https://github.com/CAMMA-public/cholect50/blob/master/docs/README-Downloads.md
</p>

<h2>Action Triplet Format: &lt;Instrument, Verb, Target&gt;</h2>
<p>
  Each frame includes up to <strong>three simultaneous triplets</strong>, one per surgical trocar. 
  Each triplet follows the format:
</p>
<pre><code>&lt;Instrument, Verb, Target&gt;</code></pre>
<p>
  Below is a breakdown of each component:
</p>

<h3>Instrument (I)</h3>
<p>Tool being used in the action:</p>
<ul>
  <li>grasper</li>
  <li>bipolar</li>
  <li>scissors</li>
  <li>hook</li>
  <li>clipper</li>
  <li>irrigator</li>
  <li>specimen_bag</li>
  <li>null_instrument <em>(if no tool is present)</em></li>
</ul>

<h3>Verb (V)</h3>
<p>Surgical action or gesture:</p>
<ul>
  <li>grasps</li>
  <li>cuts</li>
  <li>coagulates</li>
  <li>clips</li>
  <li>retracts</li>
  <li>irrigates</li>
  <li>aspirates</li>
  <li>removes</li>
  <li>dissects</li>
  <li>null_verb <em>(if no action is happening)</em></li>
</ul>

<h3>Target (T)</h3>
<p>Anatomical or surgical target of the action:</p>
<ul>
  <li>gallbladder</li>
  <li>cystic_duct</li>
  <li>cystic_artery</li>
  <li>peritoneum</li>
  <li>liver</li>
  <li>hepatocystic_triangle</li>
  <li>gallbladder_wall</li>
  <li>clip</li>
  <li>adhesion</li>
  <li>null_target <em>(if no target is specified)</em></li>
</ul>

<h3>Example Triplet</h3>

![image](https://github.com/user-attachments/assets/c4495faf-7a62-4045-bc03-12cfafa2598a)

<pre><code>&lt;bipolar, coagulate, liver&gt;</code></pre>
<p>
  This indicates the <strong>bipolar</strong> is being used to <strong>coagulate</strong> 
  the <strong>liver</strong> in that frame.
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


<h3>Accuracy Comparison</h3>

<p>
  The table below compares mAP scores for <strong>component detection</strong> (I, V, T) and <strong>triplet association</strong> (IV, IT, IVT) across state-of-the-art models and our proposed approach.
</p>

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>mAP I</th>
      <th>mAP V</th>
      <th>mAP T</th>
      <th>mAP IV</th>
      <th>mAP IT</th>
      <th>mAP IVT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Tripnet</td>
      <td>74.6</td>
      <td>42.9</td>
      <td>32.2</td>
      <td>27.0</td>
      <td>28.0</td>
      <td>23.4</td>
    </tr>
    <tr>
      <td>Attention Tripnet</td>
      <td>77.1</td>
      <td>43.4</td>
      <td>30.0</td>
      <td>32.3</td>
      <td>29.7</td>
      <td>25.5</td>
    </tr>
    <tr>
      <td>RDV</td>
      <td>77.5</td>
      <td>47.5</td>
      <td>37.7</td>
      <td>39.4</td>
      <td>39.6</td>
      <td>32.7</td>
    </tr>
    <tr>
      <td>MT-FiST</td>
      <td>82.1</td>
      <td>51.5</td>
      <td>45.5</td>
      <td>37.1</td>
      <td>43.1</td>
      <td>35.8</td>
    </tr>
    <tr>
      <td><strong>Ours (Recognition)</strong></td>
      <td>74.7</td>
      <td>49.7</td>
      <td>35.6</td>
      <td>32.8</td>
      <td>27.0</td>
      <td>23.7</td>
    </tr>
    <tr>
      <td><strong>Ours (Anticipation)</strong></td>
      <td>63.6</td>
      <td>49.6</td>
      <td>38.6</td>
      <td>36.0</td>
      <td>30.9</td>
      <td><strong>29.0</strong></td>
    </tr>
  </tbody>
</table>

<p>
  <em>Table: mAP scores for individual components (Instrument, Verb, Target) and their triplet associations, comparing baseline models with our approach.</em>
</p>


Repository Contents
Folder	Description
models/	Model components
training/	Training + evaluation scripts
utils/	Metrics and helper functions


Incorporate global temporal context (e.g., SuPRA, LoViT).

Train on more diverse surgical procedures for better generalisability.

Focus on clinical accuracy before tackling real-time deployment.

MIT / Academic Use Only (adjust based on your preference)
