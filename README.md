<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
<h1>Surgeon Action Anticipation</h1>
</head>
<body>


<h2>TL;DR</h2>
<blockquote>
  I built an AI model that predicts what a surgeon is likely to do next during keyhole surgery using deep learning.<br>
  Trained on real laparoscopic videos, the model anticipates tool use, surgical actions, and target anatomy — offering potential support for surgical training and intra-operative assistance.
</blockquote>

<h2>Installation</h2>
<pre><code>git clone https://github.com/Max-Kinnear-Nock/Surgeon-Action-Anticipation.git
cd Surgeon-Action-Anticipation
pip install -r requirements.txt
</code></pre>

<h2>Running the Model</h2>
<pre><code># Train the model
python main.py --config config.yaml

# Evaluate on the test set
python evaluate.py --config config.yaml
</code></pre>
<p>Ensure you update <code>config.yaml</code> to point to your dataset location.</p>

<h2>Predicting Surgeons’ Next Actions with Deep Learning</h2>
<p>
  This project explores the use of deep learning to anticipate surgeons’ next moves during laparoscopic procedures, contributing to the field of intra-operative AI. While AI is increasingly used for medical diagnostics, its integration into the surgical workflow remains limited. This work aims to bridge that gap by modeling both current and future surgical actions from egocentric surgical video.
</p>

<h2>Dataset: CholecT50</h2>
<p>
  The <a href="https://github.com/CAMMA-public/cholect50/blob/master/docs/README-Downloads.md">CholecT50 dataset</a> contains:
</p>
<ul>
  <li>50 full-length laparoscopic cholecystectomy (LC) surgeries</li>
  <li>Each video recorded from the surgeon’s point of view</li>
  <li>Every frame annotated with:
    <ul>
      <li>Surgical phase</li>
      <li>Tool positions</li>
      <li>Action triplets</li>
    </ul>
  </li>
</ul>
<p>
  To use: Place video frames in <code>data/videos/</code> and annotations in <code>data/labels/</code>.
</p>

<h2>Action Triplet Format: &lt;Instrument, Verb, Target&gt;</h2>
<p>
  Each frame includes up to three triplets, one per surgical port.
</p>

<h3>Instrument</h3>
<ul>
  <li>grasper</li>
  <li>bipolar</li>
  <li>scissors</li>
  <li>hook</li>
  <li>clipper</li>
  <li>irrigator</li>
  <li>specimen_bag</li>
  <li>null_instrument</li>
</ul>

<h3>Verb</h3>
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
  <li>null_verb</li>
</ul>

<h3>Target</h3>
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
  <li>null_target</li>
</ul>

<h3>Example</h3>
<pre><code>&lt;bipolar, coagulates, liver&gt;</code></pre>

<h2>Model Architecture</h2>
<ul>
  <li>ResNet-50 feature extractor</li>
  <li>Transformer encoder for temporal modeling</li>
  <li>GPT-2-style decoder for anticipation</li>
  <li>MLP classification heads for Instrument, Verb, Target, and Triplet</li>
</ul>
<img src="https://github.com/user-attachments/assets/04e1fac7-0c10-45ce-9205-857467106640" alt="Model Diagram" width="600">

<h2>Methodology</h2>

<h3>Data Split</h3>
<ul>
  <li>The official split is: 70% training, 10% validation, and 20% test (as per the CholecT50 challenge).</li>
  <li>This version uses a custom split due to long training times.</li>
  <li>Sliding window sequences are split 70/20/10 for training, validation, and testing.</li>
</ul>

<h3>Preprocessing and Augmentation</h3>
<ul>
  <li>Resize to 250x250, center crop to 224x224</li>
  <li>Random rotation ±5°, horizontal flip</li>
  <li>Brightness, contrast, saturation ±10%</li>
  <li>Hue ±5%</li>
  <li>Normalized RGB channels with dataset mean and std</li>
  <li>Sliding window sampling for temporal context</li>
</ul>

<h3>Loss Functions</h3>
<ul>
  <li>Binary Cross-Entropy with Logits for multi-label classification</li>
  <li>Multi-channel loss with discriminative and diversity terms</li>
  <li>Total Loss = Recognition + Anticipation + 0.005 × Discriminative + 2 × Diversity</li>
</ul>

<h3>Training</h3>
<ul>
  <li>20 epochs on NVIDIA A100 GPU</li>
  <li>AdamW optimizer with learning rate: 1e-4 (ResNet), 1e-3 (others)</li>
</ul>

<h3>Evaluation</h3>
<ul>
  <li>Tested using official CholecT50 scripts</li>
  <li>Metric: mean Average Precision (mAP) for I, V, T, and triplets</li>
</ul>

<h2>Qualitative Analysis</h2>
<ul>
  <li>Video playback of last recognised and next 10 predicted frames</li>
  <li>Block graphs showing ground truth vs predicted action triplets</li>
</ul>
<img src="https://github.com/user-attachments/assets/04b8b4b2-7a3e-4ca9-9181-323e7e5797bb" alt="Triplet Visualisation" width="600">

<h3>Legend</h3>
<ul>
  <li>Each frame is read left to right</li>
  <li>Center row: model predictions</li>
  <li>Bottom row: ground truth</li>
  <li>Red line: boundary between recognition and anticipation</li>
</ul>

<h2>Accuracy Comparison</h2>
<table border="1">
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
    <tr><td>Tripnet</td><td>74.6</td><td>42.9</td><td>32.2</td><td>27.0</td><td>28.0</td><td>23.4</td></tr>
    <tr><td>Attention Tripnet</td><td>77.1</td><td>43.4</td><td>30.0</td><td>32.3</td><td>29.7</td><td>25.5</td></tr>
    <tr><td>RDV</td><td>77.5</td><td>47.5</td><td>37.7</td><td>39.4</td><td>39.6</td><td>32.7</td></tr>
    <tr><td>MT-FiST</td><td>82.1</td><td>51.5</td><td>45.5</td><td>37.1</td><td>43.1</td><td>35.8</td></tr>
    <tr><td>Ours (Recognition)</td><td>74.7</td><td>49.7</td><td>35.6</td><td>32.8</td><td>27.0</td><td>23.7</td></tr>
    <tr><td>Ours (Anticipation)</td><td>63.6</td><td>49.6</td><td>38.6</td><td>36.0</td><td>30.9</td><td>29.0</td></tr>
  </tbody>
</table>

<h2>Repository Contents</h2>
<table border="1">
  <tr><th>Folder</th><th>Description</th></tr>
  <tr><td>models/</td><td>Model components</td></tr>
  <tr><td>utils/</td><td>Metrics and helper functions</td></tr>
</table>

<h2>Acknowledgements</h2>
<ul>
  <li><a href="https://arxiv.org/html/2403.06200v1">Temporal Action Transformer</a></li>
  <li><a href="https://pubmed.ncbi.nlm.nih.gov/37498758/">PubMed study on surgical AI</a></li>
  <li><a href="https://ai.meta.com/blog/anticipative-video-transformer-improving-ais-ability-to-predict-whats-next-in-a-video/">Meta AI — AVT blog</a></li>
  <li><a href="https://arxiv.org/abs/2109.03223">CholecT50 Challenge</a></li>
</ul>

<h2>Future Work</h2>
<ul>
  <li>Integrate global temporal context (e.g. SuPRA, LoViT)</li>
  <li>Train on more diverse surgical procedures for better generalisation</li>
  <li>Focus on clinical accuracy before real-time deployment</li>
</ul>

<h2>License</h2>
<p>
  MIT / Academic Use Only — please contact for commercial or clinical use.
</p>

</body>
</html>
