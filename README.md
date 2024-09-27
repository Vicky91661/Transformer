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
- Main Advantage of using transformer is : Word Token it can process parallaly
- But the Drawback is : Lack of sequential structure of the words {Order}
- Eg -  
     1. Lion kills tiger.
     2. Tiger kills lion.
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

### Normalization 
- Standard Scaling
- What is the advantages of uisng Normalization ?
    - Improved Training Stability => Not happens vanishing and exploding  Gradient Problem.
    - Faster Convergence.
    - Back Propagation - Stable Update
- alpha and beta - Learned scale  and Shift parameters
- Uses Layer Normalization

### How to Calculate the Alpha and Beta
1. Initilalized the Alpha and beta with some value
2. Compute the mean and varianece(Sigma square)
3. Normalized the input.
4. Scale and Shift.

## encoder Architecture [ Research Paper]
in reasearch paper they have used 6 Encoder. Inside 1 Encoder ->
1. input 
2. Text Embedding( Embedding vector = 512 ) + Positional Encoding
3. Multi-Head attension (8 layer)
4. Layer Normalization
5. Feed Forward NN (512 Hidden Node)

    Q = 64, K = 64, V = 64

### Residual Connection
- It's a skip connection neural network. 
- Why it used 
    - Addressing the Vanishing Gradient Problem
- How it helps in Addressing the Vanishing Gradient Problem
    - Residual Connection create a short paths for gradient to flow directly through the network. Because of this Gradient remains sufficiently large.
    - It improves the convergence. It will be faster.
    - It enables training of deeper networks.

### Why Feed Forward NN (ANN)
- It Add Non Lineararity
- Self Attension Captures the relationship between the tokens. It processes the relationship in such a way that each token can attend to every other token.
Now, This feed Forward Network based on the token, each process each token independently. this helps in tranforming these representation furthers and allows the model to learn richer representation.
- which makes more deeper learning.

## Decoder in Transformers
- The transformer decoder is responsible for generating the output sequence one token at a time, using the encoder's output and the previously generated tokens.
### 3 main Component in Decoder
1. Masked Multi Head Self Attension
2. Multi Head Attension (Encoder Decoder Attension)
3. Feed Forward Neural n/w

####  Masked Multi Head Self Attension
1. Input Embedding and  Positional Embedding
2. Linear Projection for Q,K,V
3. Scaled DOt Product Attension
4. Mask Application => Try to understand this importance
5. Multi-Head Attension
6. Concatination and Final Linear Projection
7. Residula Connection and Layer Normalization

- Output Shifted Right - Zero Padding to the right side

        input sequence              Output Sequence
       [4 5 6 7]                  [1,2,3] => Make right side zero padding [1,2,3,0]

        Output Embedding 
        [
        [0.1,0.2,0.3,0.4],
        [0.5,0.6,0.7,0.8],
        [0.9,1.0,1.1,1.2],
        [0.0,0.0,0.0,0.0]
        ]


        output Embedding + Positional Embedding = output Embedding (Suppose positional Embedding is null maxtrix)
- Linear Projection for Q,K and V
    - Create query(Q), Key(K) and Value(V) Vectors
    - WQ = WK = WV  = I
    - Q = Output Embedding * WQ = Output Embedding 
    - K = Output Embedding * WK = Output Embedding 
    - V = Output Embedding * WV = Output Embedding 
     Q= K = V  = Output Embedding
- Scaled Dot Product Attention Calculation
    - Scores = (Q*K')/Square_root(dk)    || dk = 4
    - Scores =  
                 [
                    [0.3,0.7,1.1,0.0],
                    [0.7,1.9,3.1,0.0],
                    [1.1,3.1,5.1,0.0],
                    [0.0,0.0,0.0,0.0]
                ]
- Masked Application
    - Masking in the Transformer achitecture is essential for several reasons. It helps manage the structure of the sequence being processes ans ensures the model behaves correctly during traininig and inference.
    - It helps managing the structure of the seqences being processed and ensures the models behaves correctly during training and Inferencing.
    - Reason for masked application
        - Handing vaiable length sequences with pad masking.
        - To handle sequence of different length in batch
        - To ensure that padding tokens, Which are added to amke sequences of uniform length, do not affect the model prediction.
        - since we are using zero right padding, it will influence the attention mechanism. This will intern lead to incorrect and bias predictions. That is a reason we'll do masking.
        - Two type of masking  
            1. Padding Mask 
            2. Look Ahead Mask
### Padding Mask
- [4,5,0] ( ZERO IS PADDING ) -> [1,1,0] ( PADDING MASK )
### Look ahead Mask
- Maintain Auto Regression Property
- To ensure that each position in the decoder output seqnece can only attens to previous position, but not future position.
- Sequence -> Language Modelling, Translation. 
### Combined Mask
### Masked Score
- Masked Score = Score * Combined Mask
### Softmax
- Score = softmax(Masked Scores)
## Multi-Head Attension in Decoder (Encoder and Decoder Multi Head Attension)
- Three input is going to the Multi-Head Attension of Decoder
- 2 input as => Key(K),Value(V) are comming from the Encoder part 
- 1 Input as => Query(Q) is coming from the Masked multi head Attension from Decoder Part
- These are to be used by each decoder in its "encoder-decoder" attension layer =>  THis helps the decoder to focus on appropriate places in the input sequence.


## The Final Linear and Softmax Layer
![plot](./images/plot.png)
### Linear Layer
- The Linear Layer is a simple fully connected neaural n/w that projects the vectors produced by the stack of decoders.
- It generates a very large vector , Also known as logits vector.
- Each and every block in the logits vector, it corresponds to a score of a unique word.
- If model has 10,000 vocabulary => then Logits Vectors has 10,000 cells wide
## Softmax
- Why Softmax Layer is used ?
    - It is used for multi class classification.
    - In Case of multi class classification , as soon as logits vector get passed. Every vector is going to give a log probability.
    - The Output of the softmax is log_probs.
- The softmax layer turns those scores into probabilities. The call with the highest probability is choosen and the word associated with it is produced as the output. => At that specific time stamp

