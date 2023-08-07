import streamlit as st

from utils import my_set_page

my_set_page()

st.markdown('# Customized Data Cultivation')#

with st.expander('📝 Rule Filter', expanded=False):
    st.write('''
#### Interactive Modular Rule Filter

Building a rule filter for pretrain corpus is a routine in state-of-the-art LLMs. A heuristic rule filter could preliminarily filter undesirable content efficiently.
The heuristic ideas of building rules range from text length, punctuation, special token, blocklist, and language model perplexity. 
However, no rule sets can always be valid on various data sources and languages.
Corpus from different sources could vary in quality, style, format, template, and meta information. 
Filter rules in the book field may emphasize removing structure information among high-quality content. On the contrary, when handling documents from the massive web, rules would take more attention to inspect the content quality.
The essential processes in building and improving the rules are manually concluding patterns to distinguish high- and low-quality texts and adjusting a single heuristic by examining the hit samples.

We design functions in the Interactive Modular Rule Filter module according to the above intuitions.
A user builds a rule pipeline by interactive editing and connecting rule cells referring to the patterns heuristic summarized from random samples displayed.
A rule cell could be initiated with the predefined heuristic. And the user could also customize a heuristic function and add to the predefined pool by typing Python code. 
Each rule cell's configuration, like thresholds and string patterns, can be freely adjusted according to the inspection of the hit rate and hit case.
After building a customized rule filter pipeline, Oasis can automatically generate a corresponding script according to settings and run the rule filter in the background.
    ''')


with st.expander('🖨️ Neural Filter', expanded=False):
    st.write('''
    #### Debiased Model Filter
    
The original intention of the neural filter is to select high-quality content from massive web pages similar to high-quality sources like Wikipedia. The model can filter out content with non-summarizable patterns in quality aspects.
However, treating another well-known high-quality source as positive and current sources as negative samples could lead the model bias toward the high-quality source, affecting the filtered data's quantity and diversity. Refinedweb even abandoned this process due to scruple the adverse effects of undesirable biases. 

To overcome the bias issue, we propose a negative-centric dataset-building method for neural filter training, which gathers positive samples majority from rule-filtered texts in the current source, and most negative samples by heuristic contaminate positive samples.
The predefined text contamination rule focuses on coherence and readability, including shuffling, replacing, inserting, and deleting in word, span, and sentence levels. 
The perplexities from the statistical language model may perceive these undesirable low-quality contents. However, the perplexity metric is susceptible to low-frequency special tokens and bias to the training corpus (usually Wikipedia). We only use the perplexity to find extremely low-quality content that constitutes a part of negative samples.
We model these quality patterns by a neural filter with a stronger generalization ability like BERT. The finetuned BERT predicts scores for the text quality of every rule-filtered document. Further drop documents according to quality score under the threshold.

The Debiased Model Filter module provides a management panel for the quality classification dataset. 
Users can adjust the constitution of positive and negative samples. 
As well as customize text contamination rules according to edit feedback and set perplexity quantile for finding extremely low-quality content with case inspection.
Moreover, the dataset for neural classifier training could be further evolved by adding evaluated texts by humans or GPT-4. 
After building a quality classification dataset, Oasis can generate corresponding scripts through parameter settings on the interface and run in the background with one click for the neural filter training and the running process.
        ''')

with st.expander('📑 Deduplication', expanded=False):
    st.write('''
        #### Adaptative Document Deduplication}

Repetitive documents in pretrain corpus would harm the LLM's generalization ability on various downstream tasks. 
Massive deduplication among documents is theoretic $O(n^2)$ time complexity. 
The Local Sensitive Hash algorithm approximates document similarity and reduces the time complexity. And the cost is increasing the memory requirements to store hash collision. Large-scale fuzzy deduplication is infeasible with limited resources. 

$Pr(d_i, d_j | Jaccard(d_i, d_j)=s_(i,j)) = 1 - (1 - s^{b}_{i,j})^r$

To this end, we reduce the memory requirement of the LSH deduplication algorithm for adapting to customized hardware by adjusting $r$ in the conditional probability formula.
The system predicts the maximum $r$ according to the user's configuration in corpus size and memory size. Since a smaller $r$ will lead to a lower collision probability, the system also suggests the running times according to the Jaccrad threshold and the expectation duplication recall.

Although document-level deduplication could improve the diversity of the cultivated dataset, it could also aggressively decrease the quantity.
Our Adaptative Document Deduplication module also provides an interface to visualize the duplicated documents in a graph to render options for users to tradeoffs between the removal rate and quantity.
            
            ''')

