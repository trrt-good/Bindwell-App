

# **BINDWELL: The intelligent approach to therapeutic development, powered by AI**

**Inspiration**
---------------

The exorbitant cost of prescription drugs in the US has become a crippling burden for millions of Americans. In 2020, the average American spent over $1,200 on prescription medications [1], with many more forced to choose between life-saving treatments and basic necessities. The consequences are dire: a staggering 1 in 5 Americans cannot afford their medications, leading to devastating health outcomes and even death [2]. The root of this crisis lies in the staggering cost of bringing a drug to market in the US - a whopping $2 billion and 15 years on average [3]. A significant portion of this expenditure is squandered on the inefficient and time-consuming process of high-throughput screening in drug discovery. By revolutionizing this process, we can reduce the financial burden on patients and make life-changing treatments more accessible. Bindwell was born from this urgent need, driven by the conviction that AI can transform the drug discovery process, making it faster, cheaper, and more effective.

**What is Bindwell?**
---------------------

MAKE MORE CLEAR WHAT MAKES BINDWELL UNIQUE AND WHAT'S NEW ABOUT AFFINITYLM. GO MORE INTO DEPTH ABOUT TRADITIONAL METHODS.

Bindwell is a virtual screening platform that accelerates the drug discovery process using our model, AffinityLM. This AI-powered model predicts the binding affinity of small molecules to target proteins, outperforming traditional methods in speed and accuracy. Rigorously tested against industry benchmarks, AffinityLM has consistently demonstrated superior performance, making it an invaluable tool for researchers in the pharmaceutical industry.

Here's a simple breakdown of how Bindwell works:

1. Researchers input their target protein (enzyme, receptor, hormone, etc) into the Bindwell platform as a FASTA sequence.
2. Bindwell screens a large chemical database, using AffinityLM to predict the binding affinity of each drug to the target protein.
3. The platform ranks the drugs based on their predicted binding affinity and returns the top 25 candidates.
4. The entire process takes less than a second, providing researchers with a list of the most promising drug candidates for further testing.

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
