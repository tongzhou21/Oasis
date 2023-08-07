import streamlit as st

from utils import my_set_page

my_set_page()

st.markdown('# 🔍 Holistic Data Assessment')#

with st.expander('🔬 Instance Inspection', expanded=False):
    st.write('''
        #### Local Quality Evaluation

In this module, we pay attention to a document's fluency, readability, and coherence by the judgment of humans or GPT-4.
Due to the high consumption of the human inspection process, we only provide 'High' and 'Low' two quality options in the user-friendly human evaluation interface.
It displays the statistics of manually labeled quality conditions in real time.

SOTA LLMs like GPT-4 are shown to have sufficient ability to score a document in multiple aspects to reflect the overall quality.
We provide predefined prompts for quality assessment, which achieve more than 95\% consistency with human opinions. The system also supports customized prompts for diverse demands. 
Moreover, the local quality evaluation samples can be mixed in quality classification datasets to evolve the neural filter.

            ''')

with st.expander('📊 Metrics Visualization', expanded=False):
    st.write('''
        #### Global Distribution Assessment
Apart from the local document perspective, the global view of the corpus in statistical distribution can also reflect the broadly defined quality.

Oasis adopts six metrics to assess the corpus in heuristics from a randomly sampled subset of data.

* Lexical Diversity Distribution: We calculate each document's Measure of Textual Lexical Diversity (MTLD) score to reflect the lexical diversity and plot the frequency histogram to get the overall perspective.

* Task2Vec Diversity Coefficient: The task2vec diversity coefficient is proven to have a high correlation with human's intuitive diversity of corpus. We sample batches of text and display the calculated overall score. 

* Semantic Diversity Distribution: We get all sampled documents' global semantic vector by BERT and calculate the cosine similarity of each pair of documents to plot the frequency histogram.

* Topic Diversity Distribution: We cluster the sampled documents by global vector and calculate the similarity of centroid vectors among clusters to reflect the overall topic diversity.

* Knowledge Density and Diversity: We inspect the knowledge view of the corpus by counting the different entities that occur. And the density means the entities count normalized by word count. Diversity means the semantic similarity of all emerged entities.

* Similarity to Wikipedia Distribution: Jansen et al., shows the Kenlm model's perplexity on target source could reflect the approximation of the Kenlm model's training source. We train a Kenlm model on Wikipedia and plot the perplexity distribution to inspect the extent of corpus bias in Wikipedia.

These metrics can be displayed on a single page and overlay multiple corpora for convenient visual comparison.
            ''')

