## Problem with Attension based Encoder and Decoder
1. Parallely we cannot send all teh words in a sentence -> Scalability Problem.
In huge Dataset scalability WRT Trainig is a problem.



# Transfomer
ALL the words wiil be paralley sent to encoder.
In Transfomer more the dataset increses the

### Contextual Embedding
- Example : My Name is Vicky and I play cricket.
- Embedding Layer - Word2Vec
- but we should try to get vector whenever we have longer sentences based on the relationship with other words. This is known as contextual Embedding.
- this is done suing self attention
- Because of that transformer works better.

### Roadmap for understand the Transformer

1. why Transformer
2. Architecture of Transformer
3. Self Attension - Q,K,V
4. Positional Embedding
5. Multi Head Attension
6. Combining the working of Transformers


## Architecture of Transformer
- It also follow Encoder and Decoder achitecture.
- In Encoder and Decoder part multiple encoders and decoders are present.
### Inside One Encoder
1. Self Attension Layer
2. Feed Forward Neural Network

Output of the Feed Forword neural n/w is known as Contextual Vector.
### Inside One Decoder
1. Self Attension Layer
2. Encoder and Decoder Attension
3. Feed Forward Neural Network

## Self Attension Layer at Higher and Details Level
Self-attension, also Known as scaled dot-product attension, is a crucial mechanism in the tranformer architecture that allow the model to weight the importance of different tokens in the input sequence relative to each other.

Q, K , V vector 

1. Inputs
Qeries, Keys and Values vectors

### Query vectors(Q)
Role : Query Vectors repreent the token for which we are calculating the attention. They help determine the importance of other tokens in the context of the current token.

- IMPORTANCE
1. Focus Determination : Queries help the model decide which parts of the sequence to focus on for each specific token. By Calculating the dot product between a query vector and all key vectors, the model assesses how much attention to give to each token relative to the current token.

2. Contextual Understanding : Queries contribute to understanding the relationship between the current token and the rest of the sequnce, which is essential for capturing dependencies and contex.

### Key Vectors(K)
Role : Key vectors represent all the  tokens in the seqence and are used to compare with the query vectors to calculate attenison scores.

Importance :
1. Relevance Measurement : Keys are compared with queries to measure the relevance or compatibility of each toekn with current token. This comparson helps in determining how much attension each tken should receive.
2. Information Retrieval : Keys play a critical role in retrieving the most relevant information from the sequence by providing a basis for the attention mechanism to compute similarity scores.

### Value Vectors (V):
Role : Value vectors hold the actual information that will be aggregated to form the output of the attention mechanism.

Importance :
1. Information Aggregation : Values contains the data that will be weighted by the attention scores. The weighted sum of values forms the output of the self attension mechanism, which is then passed on to the next layers in the network.
2. Context Preservation: By Weighting the values according to the attension scores, the model preserves and aggregates relevant context from teh entire sequence, which is crucial for tasks like translation, summarization, and more. 


