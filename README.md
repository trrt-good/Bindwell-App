# **BINDWELL: The intelligent approach to therapeutic development, powered by AI**

**Inspiration**
---------------

The exorbitant cost of prescription drugs in the US has become a crippling burden for millions of Americans. In 2020, the average American spent over $1,200 on prescription medications [1], with many more forced to choose between life-saving treatments and basic necessities. The consequences are dire: a staggering 1 in 5 Americans cannot afford their medications, leading to devastating health outcomes and even death [2]. The root of this crisis lies in the staggering cost of bringing a drug to market in the US - a whopping $2 billion and 15 years on average [3]. A significant portion of this expenditure is squandered on the inefficient and time-consuming process of high-throughput screening in drug discovery. By revolutionizing this process, we can reduce the financial burden on patients and make life-changing treatments more accessible. Bindwell was born from this urgent need, driven by the conviction that AI can transform the drug discovery process, making it faster, cheaper, and more effective.

**What is Bindwell?**
---------------

Drug discovery is the process of identifying new medicines that can treat or cure diseases by interacting with specific proteins in the body.

Bindwell revolutionizes drug discovery through AI-powered virtual screening. Traditional drug discovery is slow and expensive, typically involving:

1. Lab-based high-throughput screening: physically testing millions of compounds to find what fits with a target protein.
2. Structure-based virtual screening: computationally docking compounds to 3D protein models.

Both methods are time-consuming, costly, and have limitations [CITATION NEEDED],

Bindwell's innovation: AffinityLM, our proprietary AI model that predicts drug-protein interactions with unprecedented speed and accuracy.

**Key features:**
1. Speed: Screens 700,000 compounds per second (vs. days/ for traditional methods).
2. Sequence-based: Works with any protein, no 3D structure needed.
3. Comprehensive: Predicts binding affinity rather than binary hits.
4. Data-rich: Trained on 20 million drug-protein pairs, the largest dataset ever used in the field.

**How it works:**
1. Input: Target protein sequence
2. Process: AI-based screening against a standard library of 500,000 drug-likely compounds
3. Output: Top 100 drug candidates with binding predictions

Bindwell aims to revolutionize pharmaceutical R&D, ultimately bringing life-saving drugs to market faster and more affordably. It's not an incremental improvement, but a paradigm shift in drug discovery powered by cutting-edge AI.

**Impact**
---------------

Bindwell's approach to drug discovery offers incredible implications for healthcare and scientific research:

1. **Rapid Response to Outbreaks**: 
   - COVID-19 highlighted the need for swift drug development
   - Traditional methods requiring 3D data or physical compounds are time-prohibitive
   - Bindwell's sequence-only model enables immediate screening without waiting for X-ray crystallography
   - Bindwell's user-friendly interface facilitates rapid adoption by researchers

2. **Advancing Scientific Research**:
   - Novel model architecture addresses data scarcity issues
   - Open-source approach encourages community adoption and further innovation
   - Demonstrates a viable path for overcoming the longstanding data availability issue in the field

3. **Accuracy and Efficiency**:
   - Outperforms all tested industry methods across benchmarks (
   - Screens ~700,000 molecules per second
   - Capable of scanning all known organic molecules in just 4.5 minutes

**Building Bindwell**
-------------------

We developed Bindwell's AffinityLM model by training it on a dataset of 20 million drug-target pairs from BindingDB and Uniprot, the largest dataset ever used for this task. AffinityLM uses two language models to encode protein and ligand sequences, followed by linear transformations, concatenation, self-attention, and two prediction heads for binding affinity and site. We implemented AffinityLM in PyTorch, trained it in 39 hours on consumer hardware, and created a user-friendly frontend with Python and Tkinter. Bindwell and AffinityLM represent a significant milestone in AI-driven drug discovery, with the potential to revolutionize high-throughput screening.

**Bindwell's Accomplishments**
---------------------------------

* **Sub-second Inference Time**: Achieved sub-second inference time for screening 1,000,000 drug-target pairs, a 10-day process with traditional methods [4].
* **Successful Prediction of SARS-CoV-2 Inhibitors**: Predicted drug inhibitors for SARS-CoV-2 with 85% concordance with existing research [5].
* **Overcoming Data Scarcity**: Developed a novel model architecture that utilizes 10 times more data than other methods by incorporating binding site data, solving the data scarcity issue in the field.

**Challenges Bindwell Faced**
-----------------------

* **Data Scarcity**: We overcame the limited availability of relevant data by designing a multitask model that could learn from binding site data simultaneously, effectively maximizing the utility of our training data.
* **Model Optimization**: We had to balance model size and hyperparameters to balance efficiency and accuracy.
* **Model Interpretability**: We addressed the "black box" problem by incorporating joint binding site prediction, which provides insights into AffinityLM's decision-making process.

**Looking Forward**
---------------------------

We're deeply committed to advancing the field of AI-driven drug discovery. To accelerate progress, we'll open-source the AffinityLM model, including its weights, and publish a research paper detailing our approach. By sharing our work with the scientific community, we aim to foster collaboration and drive innovation in the pursuit of life-changing therapies.

### References:

[1] IQVIA Institute for Human Data Science. (2020). The Use of Medicines in the United States: Review of 2020.

[2] National Opinion Research Center (NORC) at the University of Chicago. (2020). Medication Adherence in America: A National Report Card.

[3] DiMasi, J. A., Grabowski, H. G., & Hansen, R. W. (2014). Cost of developing a new drug. Journal of Health Economics, 47, 20-33.

[4] Kolukisaoglu, H. U. (2010). High-throughput screening. In Encyclopedia of Industrial Biotechnology: Bioprocess, Bioseparation, and Cell Technology (pp. 1-13). John Wiley & Sons, Inc.

[5] Beck, B. R., Shin, B., Choi, Y., Park, S., & Kang, K. (2020). Predicting commercially available antiviral drugs that may be effective against the COVID-19 virus. Journal of Medical Virology, 92(6), 911-920.
