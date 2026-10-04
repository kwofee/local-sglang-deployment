# Chapter 1: Introduction to LLMs

## 1. Learning objectives

Welcome to the first chapter of the *zero-to-sglang* course. As the course's starting point, this chapter builds a high-level understanding of Large Language Models (LLMs), providing a foundation for later discussions of inference, GPU architecture, and inference frameworks.

After completing this chapter, you should be able to:

1. Explain the basic definition and development of LLMs, including what they can and cannot do.
2. Identify typical real-world LLM applications.
3. Understand the Transformer—the core LLM architecture—and the roles of positional encoding, multi-head attention, layer normalization, residual connections, and feed-forward networks.
4. Be familiar with foundational concepts used throughout the course: tokens, parameters, context windows, and autoregressive generation.

## 2. Definition, brief history, and limits of LLMs

**What is an LLM?** A large language model is a language model based on deep neural networks and trained on massive amounts of text. Its fundamental ability is to predict the most likely next token given some text. At sufficiently large model and data scales, this seemingly simple “next-token prediction” ability gives rise to diverse capabilities: understanding, reasoning, generation, translation, and coding. “Large” refers both to the number of parameters (from billions to hundreds of billions, or even trillions) and to the volume of training data.

- **2017:** Google introduced the Transformer architecture in [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). It replaced RNNs/CNNs with self-attention and laid the foundation for modern LLMs.
- **2018–2019:** The BERT (bidirectional encoder) and GPT (autoregressive decoder) approaches emerged. Pre-training followed by fine-tuning became the mainstream paradigm.
- **2020:** GPT-3 (175B parameters) demonstrated large models' few-shot learning ability, drawing broad attention to scaling effects.
- **2022–present:** ChatGPT sparked an application boom. Open-source models such as LLaMA, Qwen, and DeepSeek have iterated quickly, while architectures and training techniques continue to evolve (for example RoPE, GQA/MLA, RMSNorm, and SwiGLU).

**Capability boundaries.** Although powerful, an LLM is fundamentally a probabilistic model that continually predicts the next token. It therefore has clear limitations: it can produce plausible but incorrect content (hallucinations); its knowledge is limited by the training-data cutoff; its memory of text beyond its context window is limited; and it lacks genuine causal reasoning and real-time perception of the external world. Understanding these boundaries helps us use and optimize models appropriately.

## 3. Typical LLM applications

LLMs have reached many practical settings. Typical applications include the following.

### 3.1 Conversation and question answering

Conversational assistants are the traditional main arena for LLMs: a user asks a question and the model answers. From GPT-3 onward, such assistants have improved greatly—from single-modal text systems to multimodal systems that work with text, images, and audio, and from offline chat to web-connected search. To some extent, conversational assistants now occupy part of the role once held by search engines: when people have a question, they may ask AI instead of searching in a browser.

### 3.2 Text generation

Content generation and creative work are among the areas where LLMs have most visibly improved productivity. LLM-assisted writing and report drafting are now common; more involved uses include official-document writing, meeting minutes, and direct generation of PDFs and Word documents. Text generation is already widely used.

LLMs also shine in translation and are increasingly replacing traditional machine translation. The “flash” models released by various vendors are becoming important tools in translation.

### 3.3 Coding

Claude Code, Codex, and Cursor are among the most popular recent AI programming tools. Domestic products such as QCode, ZCode, and CodeBuddy are also competing in the market. These “vibe coding” tools allow non-specialists to create something with a single sentence while reducing programmers' workload. They are becoming so popular that not having used them can feel like falling behind.

All of these applications depend on efficient model **inference**. This is exactly what later chapters of the course—inference pipelines, GPUs, and inference frameworks—focus on solving.

### 3.4 The foundations of AI applications: inference cost and latency

These applications are centered on large models. In practice, using an AI tool means making use of a large model's output, and that output depends on inference. Inference cost directly affects the cost of using a model, while inference latency directly affects user experience. Both are therefore crucial.

The autoregressive nature of LLM generation means memory requirements grow linearly with input-sequence length and batch size. In high-concurrency scenarios, more than 70% of GPU memory can be occupied by the KV cache (introduced later), limiting the concurrency and context length of a single GPU. Recalculation caused by cache eviction raises costs further. An NVIDIA estimate cited by the source says that inference clusters without KV-cache offloading optimization can incur 2–3× higher per-token generation cost from repeated computation, and as much as 3.5× during peak concurrency. In one medium-sized enterprise AI customer-service example, extra cost from KV-cache memory bottlenecks and cache eviction accounted for 78% of monthly operating cost.

Latency directly affects the experience. Time to first token (TTFT) and time per output token (TPOT) determine how responsive an interaction feels. Meeting a service-level target at or below human reading speed (≤150 ms/token) has become a key deployment constraint.

These are the problems efficient inference tools address. Efficient inference frameworks exist to solve the increasingly severe cost and latency bottlenecks created by explosive application growth, helping LLMs move from technically feasible to commercially viable.

## 4. Core architecture: Transformer

The Transformer originated in 2017, when a Google research team introduced it in [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). Its key innovation was the **self-attention mechanism**, which discarded traditional recurrent neural network (RNN) and convolutional neural network (CNN) structures. Self-attention lets a model compute sequence data in parallel, substantially improving efficiency and addressing long-range dependencies.

> Figure 1: Overall Transformer architecture (in the original chapter).

The figure shows Transformer blocks: the **encoder block** is on the left and the **decoder block** on the right. Stacking encoder and decoder blocks *N* times forms the Transformer architecture.

### 4.1 Positional encoding — sinusoidal positional encoding

$$
\begin{align*}
PE_{(pos,2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\
PE_{(pos,2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\end{align*}
$$

**Variables:**

- $pos$: a token's position in the sequence (0, 1, 2, ..., N−1)
- $i$: dimension index
- $d_{model}$: model embedding dimension (512 in the paper)
- $10000$: base frequency (configurable)

After calculating positional encodings, add them directly to the token embeddings: $X = Token + PE(pos)$.

Sinusoidal positional encoding is a key Transformer design for introducing **position information** into a sequence. Because the core self-attention mechanism has **no inherent awareness of input order**, feeding it only word vectors would treat “I love you” and “You love me” as the same set. Position information must therefore be explicitly added to the input. Sinusoidal positional encoding requires no training: it uses fixed formulas based on sine and cosine functions at different frequencies to generate a unique encoding vector for each sequence position, while also representing relative-position relationships.

For position $pos$ (counting from 0) and a vector dimension $d$ (total dimension $d_{model}$), define the paired-dimension index $i = 0, 1, \dots, \frac{d_{model}}{2} - 1$. Use sine for even dimensions ($d = 2i$) and cosine for odd dimensions ($d = 2i+1$). The even and odd dimensions in the same pair **share the same frequency base**:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)
$$

Here, $10000^{2i / d_{model}}$ determines the wavelength in each dimension, creating a multi-scale encoding system. Low dimensions (small $i$) have high frequency and finely distinguish neighboring positions; high dimensions (large $i$) have low frequency and cover more distant relative relationships. Each position ultimately receives a vector with the same dimension as its word embedding, and the two are added to form the model input.

This design has several advantages. First, it is fully deterministic and needs **no additional parameters**, avoiding greater model complexity or overfitting of learned positional embeddings. Second, trigonometric periodicity constrains values to $[-1, 1]$, making them numerically stable and easy to add to token embeddings. More importantly, it **naturally models relative positions**: for any fixed offset $k$, the encoding at $pos+k$ can be expressed as a linear transformation of the encoding at $pos$ that depends only on $k$. This makes it easier for self-attention to learn relative rather than merely absolute positions, which is especially useful for variable-length sequences and local dependencies.

Sinusoidal encoding was introduced in *Attention Is All You Need* as the original Transformer's standard positional-encoding method. Learned positional embeddings and relative positional encodings have since appeared, but sinusoidal encoding remains widely used in sequence modeling and is a classic way to understand positional encoding.

### 4.2 Multi-head attention

> Figure 2: Multi-head attention (in the original chapter).

Attention is modeled after human attention: when viewing an image, people do not distribute attention evenly across every corner; they selectively observe salient parts. Similarly, attention assigns larger weights to important parts of its input. At its core, attention is a weighted sum.

Multi-head attention is a core Transformer innovation. By running several attention heads in parallel, it lets a model attend simultaneously to multiple dependencies in a sequence from different perspectives and semantic levels, greatly improving its ability to model complex patterns. We begin with the method and limitations of a single attention head.

#### 4.2.1 Single-head attention: method and limitations

Single-head attention uses scaled dot-product attention. Given input sequence $X$ with shape $[batchSize, seqLen, d_{model}]$, project it with three weight matrices to obtain queries (Q), keys (K), and values (V):

$$
Q = X W^Q, \quad K = X W^K, \quad V = X W^V
$$

The shapes of $W^Q$, $W^K$, and $W^V$ are all $[d_{model}, d_{model}]$, so Q, K, and V remain $[batchSize, seqLen, d_{model}]$. The attention output is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)V
$$

Here $d_k = d_{model}$. The scaling factor $\sqrt{d_k}$ prevents dot products from becoming so large that gradients enter a saturated region.

The limitation of a single head is that it computes only one query–key–value relationship—as if observing with only one pair of eyes. It can struggle to capture syntax, semantic associations, and long-range dependencies simultaneously, and its attention distribution may be too diffuse to focus on multiple important subspaces. The common solution is to repeat attention several times, have each head learn a different subspace representation, and then combine the results.

#### 4.2.2 The design idea of multi-head attention

Multi-head attention splits $d_{model}$-dimensional queries, keys, and values into $h$ independent heads. Each head performs attention in parallel in a lower-dimensional space, where $d_k = d_{model}/h$. This lets the model jointly extract information from several representation subspaces. Each head has its own projection matrices and can attend to different feature types. For example—only as an intuition, not a known division of labor—some heads may focus on local syntax while others capture long-distance semantic dependencies. Internally, the model is a series of floating-point values, so we cannot reliably assign a concrete role to each head; practice nevertheless shows the method is useful.

#### 4.2.3 Detailed multi-head-attention computation

**Step 1: Split into heads** (assuming Q, K, and V have already been linearly projected)

For Q, K, and V, all shaped $[batchSize, seqLen, d_{model}]$, split into $h$ heads. Reshape and transpose to move the head dimension forward:

```python
Q = Q.reshape(batch_size, seq_len, h, d_k)   # [bs, seq_len, h, d_k]
Q = Q.transpose(1, 2)                        # [bs, h, seq_len, d_k]
```

After transposition, Q, K, and V have shape $[batchSize, h, seqLen, d_k]$. Each head can then be processed independently and efficiently in batches.

**Step 2: Compute attention for every head in parallel**

For head $i$, independently apply scaled dot-product attention:

$$
\text{Head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right)V_i
$$

$Q_i$, $K_i$, and $V_i$ are each $[batchSize, seqLen, d_k]$, so $Q_iK_i^T$ is $[batchSize, seqLen, seqLen]$: attention scores between every pair of positions within that head. The head output is also $[batchSize, seqLen, d_k]$.

**Step 3: Concatenate and apply the final linear transformation**

Concatenate all heads along the head dimension to restore the original dimensionality:

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{Head}_1, ..., \text{Head}_h)
$$

The concatenated shape is $[batchSize, seqLen, h \times d_k] = [batchSize, seqLen, d_{model}]$. Finally, apply an output projection matrix $W^O$, shaped $[d_{model}, d_{model}]$:

$$
\text{Output} = \text{MultiHead}(Q,K,V) W^O
$$

The output has the same dimension as the input, making residual connections and subsequent operations straightforward.

#### 4.2.4 Parameters in the original paper

In *Attention Is All You Need*, multi-head attention used:

- Model dimension: $d_{model} = 512$
- Number of heads: $h = 8$
- Per-head dimensions: $d_k = d_v = d_{model}/h = 64$
- Scaling factor: $\sqrt{d_k} = \sqrt{64} = 8$

This configuration keeps computation close to that of a single attention head (each head works in a lower-dimensional space, so total compute is broadly unchanged) while substantially improving representation capacity.

### 4.3 Layer normalization and residual connections

#### 4.3.1 What is normalization?

**Normalization scales data according to specific rules** so it falls within a common standard range or distribution. In deep learning, it generally means transforming intermediate neural-network activations or weights to stabilize training and speed convergence.

The probability-theory normalization formula is $x_{norm} = (x - \mu) / \sigma$, where $\mu$ and $\sigma$ are the original data's mean and standard deviation. This resets the data to a standard state: subtracting the mean and dividing by the standard deviation produces a distribution with **mean zero and variance one**.

The original Transformer’s layer normalization is:

$$
\text{LayerNorm}(v) = \gamma \frac{v - \mu}{\sigma} + \beta
$$

Layer normalization normalizes a single sample across all neurons in the same layer. It is not fundamentally different from ordinary normalization.

1. **Compute the mean** across all neurons in the layer:

   $$
   \mu = \frac{1}{d} \sum_{i=1}^{d} v_i
   $$

2. **Compute the standard deviation:**

   $$
   \sigma = \sqrt{\frac{1}{d} \sum_{i=1}^{d} (v_i - \mu)^2 + \varepsilon}
   $$

   $\varepsilon$ is a tiny constant (such as $10^{-6}$) that prevents division by zero.

3. **Normalize:**

   $$
   \hat{v} = \frac{v - \mu}{\sigma}
   $$

4. **Apply learned parameters:**

   $$
   \text{Output} = \gamma \cdot \hat{v} + \beta
   $$

The learnable parameters $\gamma$ (scale) and $\beta$ (shift) have the same dimension as the input, allowing the model to learn the appropriate scaling and shift.

#### 4.3.2 What is a residual?

In deep learning, a **residual** specifically means a **residual connection**, also called a **skip connection**. It is a shortcut between neural-network layers that lets information bypass certain layers directly.

> Figure 3: Residual connection and layer normalization (Add & Norm), in the original chapter.

The residual formula is:

$$
Output = Input + Layer(Input)
$$

In the figure, a residual connection directly adds the data from before the multi-head-attention layer to the layer output. Abstractly, it introduces a flow of shallow-layer information, preventing information loss and growing error as networks become deeper. It continually brings forward previous information as a correction.

Mathematically, instead of forcing a network to learn the desired mapping $H(x)$ directly, residual learning asks it to learn the residual $F(x) = H(x) - x$. If a layer need not transform the input, the network only needs to learn $F(x) \approx 0$, preserving $x$. If a transformation is needed, it learns the correction to the input. In the extreme case, even if `Layer` learns poorly, $Output \approx Input$, so the result should be no worse than omitting that additional depth.

#### 4.3.3 Layer normalization together with residual connections

In the original Transformer paper, layer normalization and residual connections work together to make deep networks train stably. The original paper applies the residual connection first, then normalizes:

$$
\text{output} = \text{LayerNorm}\big(x + \text{Sublayer}(x)\big)
$$

Here, $x$ is the tensor input to the current sublayer, and $\text{Sublayer}(x)$ is the sublayer’s transformation (such as multi-head self-attention or a feed-forward network). $x + \text{Sublayer}(x)$ is the residual connection, and $\text{LayerNorm}$ is applied afterward.

This is Post-Norm, whose order is **sublayer → residual → normalization**, in contrast to the later Pre-Norm variant (normalize first, then sublayer, then residual).

**Residual connections let gradients propagate directly and retain identity-mapping capability; layer normalization standardizes the resulting distribution and prevents numerical explosion or vanishing.** Together, they make networks with 12 layers or more trainable.

### 4.4 Feed-forward networks and activation functions

> Figure 4: Feed-forward network, in the original chapter.

The original Transformer paper, *Attention Is All You Need*, uses the **ReLU** (Rectified Linear Unit) activation function in the **position-wise feed-forward network**.

#### 4.4.1 Where ReLU is applied

In every encoder and decoder layer, the feed-forward network is:

$$
\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2
$$

The **first layer** is a **linear transformation** followed by **ReLU**; the **second layer** is only a **linear transformation**, with no activation function.

#### 4.4.2 Parameter configuration

| Parameter | Value | Description |
|---|---:|---|
| **Input/output dimension** | $d_{model} = 512$ | Same as the model’s main dimension |
| **Hidden dimension** | $d_{ff} = 2048$ | Expand by 4×, then compress |
| **Activation** | $ReLU$ | Used only in the first layer |

The full process is: **512-dimensional input** → **linear layer (512 → 2048)** → **ReLU** → **linear layer (2048 → 512)** → output.

#### 4.4.3 Why use ReLU?

ReLU is **computationally efficient**. Compared with sigmoid or tanh, its derivative is simple to calculate (0 or 1).

#### 4.4.4 Basic properties of a good activation function

First, it must be **nonlinear**. No matter how deep a neural network made only of linear functions is, it reduces to a simple linear model that can only fit simple functions. Nonlinearity makes the network more complex and lets it learn complex problems.

Second, it should be **differentiable** almost everywhere in its domain, supporting gradient descent and backpropagation. Otherwise, the model cannot be trained with gradient descent. ReLU is not differentiable at $x=0$, but a subgradient can be used effectively in practice.

Third, it must be **simple and efficient to compute**. An activation function is called billions of times during training and inference, so its compute cost must remain low at scale.

### 4.5 Beyond the lesson: from these foundations to state of the art

This introductory lesson covers the most classic, fundamental Transformer components. Industry and research move quickly, and many components have been replaced by more advanced methods. The following is only a **terminology comparison**, without detailed explanation, for further self-study.

| Component / dimension | Foundation in this lesson | Evolution → state of the art (SOTA) |
| :--- | :--- | :--- |
| Overall architecture | Encoder–Decoder | → Decoder-only → MoE (Mixture of Experts) |
| Positional encoding | Sinusoidal / absolute positional encoding | → Learned positional encoding → ALiBi → RoPE → NoPE |
| Attention | MHA (multi-head attention) | → MQA → GQA → MLA → NSA / DSA → KDA |
| Attention scope | Global dense attention | → Sparse attention → Sliding-window attention → Global/local hybrid |
| Normalization method | LayerNorm | → RMSNorm → Dual normalization (Pre + Post Norm) / QK-Norm |
| Normalization placement | Post-Norm | → Pre-Norm |
| Activation | ReLU | → GeLU → Swish → GLU family (GeGLU / SwiGLU) |
| Feed-forward network | Dense FFN (Dense MLP) | → MoE FFN |
| KV handling | No cache; recomputed each time | → KV cache → KV quantization → PagedAttention |
| Positional extrapolation / long context | Fixed context length | → RoPE interpolation/extension (NTK, YaRN, etc.) → long-context architectures |
| Compute precision | FP32 | → FP16 / BF16 → FP8 → INT8 / INT4 |

## 5. Key foundational concepts

### 5.1 Prompt engineering

This is the most basic and direct interaction method. A **prompt** is an instruction or question given to AI: the input text that triggers generation. **Prompt engineering** is the systematic study and design of prompts so the AI more reliably and accurately produces the desired output.

Think of it as a document written for AI. The key is to state the background and requirements clearly, making output more controllable. When a task becomes complex and the prompt must contain large amounts of dynamic information, prompts become bloated and results can worsen.

### 5.2 Context engineering

If prompt engineering is asking a question well, context engineering prepares all the background material for the conversation. **Context** is all information supplied to an LLM during one inference run, **including prompts, conversation history, and external data**. **Context engineering** is the discipline of designing and optimizing that information to improve AI understanding and task performance.

Context engineering is not about throwing every document at an LLM. It manages the AI’s **limited working memory**—the context window—and decides which information belongs in it at a particular time. **Context engineering is the broader system design; prompt engineering writes the specific instruction within that designed framework.**

Good context engineering is important because it determines what information to give the model and when. First, a model’s context window is limited: feeding it everything can exhaust the window and dilute attention, making performance appear less capable. Second, cost matters: including all information on every inference consumes more tokens.

### 5.3 Skills

A **skill** can be understood as a **structured prompt** or a **specialist capability package for AI**. It packages the experience, workflow, and rules for a particular task into a reusable standardized module. A skill tells the model how to do something and which tools to call.

If an LLM is a person, a skill is an operating manual that lets its brain perform a specific task precisely. It addresses the gap where a model knows what to do but not exactly how. Skills are an agent’s hands and feet: its core capability units for carrying out concrete tasks.

### 5.4 Retrieval-augmented generation (RAG)

**RAG (Retrieval-Augmented Generation)** is a technique in which an LLM retrieves the most recent and relevant information from an external knowledge base before generating an answer.

It addresses two major LLM problems: stale knowledge, since training data has a cutoff date; and confidently incorrect output, or hallucination. By introducing external authoritative knowledge as evidence, RAG can substantially improve answer accuracy and reliability.

RAG is a key technique for context engineering. It retrieves information from long-term memory (such as a vector database) to fill the model’s short-term working memory (the context window).

### 5.5 Agents

If the concepts above help AI think and answer better, an **agent** lets AI begin to act. An **AI agent** is a software entity that can autonomously perceive its environment, understand goals, and execute tasks to achieve them.

It resembles a person with autonomous behavior: **autonomy** to work independently, **reactivity** to respond dynamically, and **goal-directedness** to plan actions that complete a task.

A typical agent workflow is **goal decomposition + skill invocation + result verification**. After receiving a task, it breaks it into steps, calls the appropriate skills, performs them, and checks the results.

### 5.6 Harness

A **harness** can be understood as an **agent’s runtime foundation or operating system**. It is not a specific model, but an engineering framework that gives a model its hands, feet, and nervous system.

It handles the engineering needed to let a model work: tool calls, task planning, execution scheduling, memory management, and more. A popular formula is: **model (brain) + harness (hands, feet, and nervous system) = agent**. It lets you combine plugins like building blocks to create a custom agent.

### 5.7 Other foundational concepts

**Token:** The smallest unit into which a tokenizer splits text. The model actually processes integer IDs corresponding to tokens. For DeepSeek, one token is approximately 1.6–1.7 Chinese characters.

**Embedding:** A mapping from a discrete token ID to a continuous vector representation. Embedding vectors contain semantic information and can be used for clustering and classification; document embeddings can also be used by RAG for retrieval.

**Parameters:** Learnable weights in a model. Parameter count commonly measures model scale: for example, 7M or 70B, where M means million and B means billion. A 600B-parameter model has 600 billion parameters; each parameter is a floating-point value.

**Autoregressive generation:** The model generates tokens one at a time, each time using already-generated content as input to predict the next token.

**Context window:** The maximum number of tokens a model can process at once; it determines how much text the model can see. Current flagship models commonly support 1M-token contexts, roughly enough for 1.5–2 copies of *Dream of the Red Chamber*.

## 6. Summary and quiz

### 6.1 Chapter summary

This chapter built a high-level understanding of large language models. An LLM is essentially a large neural network that predicts the next token; scaling produces diverse capabilities. LLMs have developed rapidly from the introduction of the Transformer to today’s thriving open-model ecosystem, but they retain limitations such as hallucinations and time-limited knowledge. Their core Transformer architecture comprises four major components: **positional encoding, multi-head attention, layer normalization with residual connections, and feed-forward networks**. Self-attention enables efficient parallel computation. These concepts are the foundation for later study of inference pipelines and systems optimization.

### 6.2 Quiz

1. In one sentence, state an LLM’s core ability, and explain why predicting the next token can lead to such rich capabilities.
2. Why does self-attention need positional encoding? How does sinusoidal positional encoding supply positional information?
3. Briefly describe the roles of residual connections and layer normalization.

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [DIY-LLM, Chapter 13: The basic training process of large models](https://datawhalechina.github.io/diy-llm/chapter13/chapter13_%E7%AC%AC%E5%8D%81%E4%B8%89%E7%AB%A0%E5%A4%A7%E6%A8%A1%E5%9E%8B%E7%9A%84%E5%9F%BA%E6%9C%AC%E8%AE%AD%E7%BB%83%E6%B5%81%E7%A8%8B.html)
- [Tencent Cloud FlexKV inference-cost article (DoNews)](https://www.donews.com/news/detail/4/6512672.html#1)
