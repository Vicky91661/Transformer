## Problem with Attension based Encoder and Decoder
1. Parallely we cannot send all teh words in a sentence -> Scalability Problem.
In huge Dataset scalability WRT Training is a problem.

# Transfomer
ALL the words wiil be paralley sent to encoder.
In Transfomer more the dataset increses the

### Contextual Embedding
- Example : My Name is Vicky and I play cricket.
- Embedding Layer - Word2Vec
- but we should try to get vector whenever we have longer sentences based on the relationship with other words.This is known as contextual Embedding.
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
2. Context Preservation : By Weighting the values according to the attension scores, the model preserves and aggregates relevant context from the entire sequence, which is crucial for tasks like translation, summarization, and more.

## Linear Transformation
We create Q, K,V bt multiplying the embeddig by learned weights matrices Wq, Wk and Wv.

Example : 
- Wq = Wk = Wv = I (Itentity matrix)
- THE =>  [1 0 1 0] 
- QTHE = [1 0 1 0] I = [1 0 1 0] 
- KTHE = [1 0 1 0] I = [1 0 1 0]
- VTHE = [1 0 1 0] I = [1 0 1 0]

- QTHE = KTHE = VTHE = [1 0 1 0]
- similarly
- QCAT = KCAT = VCAT = [0 1 0 1]
- similarly
- QSAT = KSAT = VSAT = [1 1 1 1]

### Compute Atension Scores

- For "THE"
- Score(QThe,KThe) = [1 0 1 0] [1 0 1 0]T = 2
- Score(QThe,KCat) = [1 0 1 0] [0 1 0 1]T = 0
- Score(QThe,KSat) = [1 0 1 0] [1 1 1 1]T = 2
- i.e. For THE, SAT is important.

#

- For "CAT"
- Score(QCat,KThe) = [0 1 0 1] [1 0 1 0]T = 0
- Score(QCat,KCat) = [0 1 0 1] [0 1 0 1]T = 2
- Score(QCat,KSat) = [0 1 0 1] [1 1 1 1]T = 2

- i.e. For CAT, SAT is important.
#
- For "SAT"
- Score(QSat,KThe) = [1 1 1 1] [1 0 1 0]T = 2
- Score(QSat,KCat) = [1 1 1 1] [0 1 0 1]T = 2
- Score(QSat,KSat) = [1 1 1 1] [1 1 1 1]T = 4
- - i.e. For SAT, The and Cat are important.
## SCALING
#### without Applying  Scaling
- Scaling is the attension Mechanism which is crucial to present the dot product from grawing too large. => Ensure stable graients during Training.

- what happens when dk is large ->
1. Gradient Exploding
2. Softmax Saturation -> Vanishing Gradient  problem
- what does the softmax([6,4]) = [0.88,0.12] means ?
- Most of the attension weight is assigned to the first key vector, very little to the second vector.
- Property of Softmax.

#### After Applying  Scaling
- After Applying Scaling The attension weights become more balanced compared to the unscaled case.
- STABILIZING TRAINING : Scaling prevent extremely large dot product, which helps in stabilbizing the gradients during backpropagation, making the training process more stable and efficient.
- PREVENTING SATURATION : By Scaling the dot products, the softmax function produces more balanced attention weights, preventing the model from focusing too heavily on a sigle token and ignoring others. 
- IMPROVED LEARNING : Balanced attension weights enable the model to learn better representations by considering multiple relevant tokens in the sequence, leading to better performance on tasks that require context understanding.
- Scaling ensures that the dot products are kept within a range that allows the softmax function to operate effectively, providing a more balanced distribution of attension weights and improving the overall learning process of the model.


### Steps to find the contextual vector
1. Find the value of Q,K,V 
2. Find the value of Wq, Wk, Wv
3. FInd the value of attention score
4. Find the scaled
5. Apply Softmax
6. calculate the weightd sum of values => Attention Head

## Self Attention with multi Heads


## Positional Encoding ( Representing Order of sequence )
- Main Advantage of using transformer is : Word Token it can process parallely
- But the Drawback is : Lack of sequential structure of the words {Order}
- Eg - 
- 1. Lion kills tiger.
- 2. Tiger kills lion.
- Both the stenences are represneted with the same vectors. it does not take care of the ordering part.
- In order to prevent that Positional Encoding is used. which represent the order of sequence.
### Types of Position Encoding
1. Sinusoidal Position Encoding
2. Learned Positionsal Encoding => Positional Encoding are learned during Training

#### Sinusoidal Position Encoding
- It used sine and cosine functions of different frequencies to create positional encodings.
- Sine and cosine functions are used because they provide a continuous and differentiable method to encode position information, which helps in training deep learning models. Their periodic nature allows the model to learn and generalize across different positions effectively, and their alternating use across dimensions helps in maintaining unique encodings for each position.

- i = even => Use sine formula
- i = odd => Use cosine formula

- "THE" Word => [0.1,0.2,0.3,0.4] => [0,1,0,1]
- "CAT" Word => [0.5,0.8,0.7,0.8] => [0.84,0.54,0.01,0.995]

- "THE" Word => [0.1,0.2,0.3,0.4] + [0,1,0,1] => GO TO SELF ATTENSION
- "CAT" Word =>[0.5,0.8,0.7,0.8] + [0.84,0.54,0.01,0.995] => GO TO SELF ATTENSION


## Layer Normalization In Transformers (Add and Normalization)
- Multi Head  Attention gives the Residuals -> Additional Signal to the layer normalization.
- Normalization
    - Batch Normalization
    - Layer Normalization


