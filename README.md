# DRAKES Sequence Generator

## Overview

DRAKES integrates sophisticated mathematical frameworks to generate DNA sequences utilizing diffusion processes, Transformer architectures, adversarial training, and reinforcement learning. The model leverages stochastic differential equations, probabilistic transitions, and optimization algorithms to ensure the generation of biologically plausible and functionally diverse DNA sequences.

## Mathematical Model

### Diffusion Process

The diffusion process models the transformation of an initial data distribution \( q(x_0) \) into a complex target distribution \( p(x_T) \) through a series of discrete time steps \( t = 1, 2, \ldots, T \). The forward diffusion process introduces noise incrementally:

\[
x_t = \sqrt{\alpha_t} \, x_{t-1} + \sqrt{1 - \alpha_t} \, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
\]

where:
- \( x_0 \sim q(x_0) \) is the initial data sample.
- \( \alpha_t \) is the noise schedule parameter at time step \( t \).
- \( \epsilon_t \) represents Gaussian noise added at each step.

The reverse diffusion process aims to reconstruct the original data by modeling the conditional distribution:

\[
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
\]

Parameters \( \mu_\theta \) and \( \Sigma_\theta \) are learned via neural networks to approximate the reverse transitions.

### Transformer Architecture

Transformers employ self-attention mechanisms to capture dependencies within sequences. The self-attention operation is defined as:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V
\]

where:
- \( Q = X W_Q \), \( K = X W_K \), and \( V = X W_V \) are the query, key, and value matrices obtained by linear transformations of the input \( X \).
- \( d_k \) is the dimensionality of the key vectors.

#### Multi-Head Attention

Multiple attention heads allow the model to focus on different representation subspaces:

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
\]
\[
\text{head}_i = \text{Attention}(Q W_{Q_i}, K W_{K_i}, V W_{V_i})
\]

where \( h \) is the number of heads, and \( W_O \) is the output projection matrix.

#### Positional Encoding

To incorporate positional information, positional encodings \( \text{PE}(pos, 2i) \) and \( \text{PE}(pos, 2i+1) \) are defined as:

\[
\text{PE}(pos, 2i) = \sin\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)
\]
\[
\text{PE}(pos, 2i+1) = \cos\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)
\]

where \( pos \) denotes the position in the sequence and \( i \) indexes the dimension.

### Gumbel-Softmax Trick

The Gumbel-Softmax enables differentiable sampling from categorical distributions, facilitating gradient-based optimization in models involving discrete variables. Given logits \( \pi \), the Gumbel-Softmax sample \( y \) is computed as:

\[
y = \text{softmax}\left( \frac{\log(\pi) + g}{\tau} \right), \quad g \sim \text{Gumbel}(0,1)
\]

where \( \tau \) is the temperature parameter controlling the smoothness of the distribution.

### Kullback-Leibler Divergence

The KL divergence quantifies the discrepancy between two probability distributions \( P \) and \( Q \):

\[
\text{KL}(P \| Q) = \sum_{x} P(x) \log\left( \frac{P(x)}{Q(x)} \right)
\]

In DRAKES, it measures the divergence between the learned transition rates \( p_\theta(x_{t-1} | x_t) \) and the true distribution \( q(x_{t-1} | x_t) \):

\[
\mathcal{L}_{\text{KL}} = \sum_{t=1}^{T} \text{KL}\left( p_\theta(x_{t-1} | x_t) \| q(x_{t-1} | x_t) \right)
\]

### Diversity Loss

To promote diversity in generated sequences, the diversity loss \( \mathcal{L}_{\text{diversity}} \) is defined based on pairwise Hamming distances \( d_H(x_i, x_j) \):

\[
\mathcal{L}_{\text{diversity}} = -\frac{1}{N(N-1)} \sum_{i \neq j} \frac{d_H(x_i, x_j)}{L}
\]

where:
- \( N \) is the batch size.
- \( L \) is the sequence length.

### Adversarial Loss

Adversarial training involves a discriminator \( D \) that distinguishes between real and generated sequences. The adversarial loss \( \mathcal{L}_{\text{adv}} \) for the generator is:

\[
\mathcal{L}_{\text{adv}} = \mathbb{E}_{x \sim P_{\text{gen}}} \left[ \log(1 - D(x)) \right]
\]

where \( P_{\text{gen}} \) is the distribution of generated sequences.

### Constraint Loss

Constraints ensure generated sequences meet biological criteria, such as GC content. The constraint loss \( \mathcal{L}_{\text{constraint}} \) comprises terms like GC content loss \( \mathcal{L}_{\text{gc}} \):

\[
\mathcal{L}_{\text{gc}} = \left( \frac{\sum_{i=1}^{L} \mathbb{I}(x_i \in \{C, G\})}{L} - \text{GC}_{\text{target}} \right)^2
\]

where \( \mathbb{I} \) is the indicator function, and \( \text{GC}_{\text{target}} \) is the desired GC content.

## Model Components

### Masked Diffusion Model

The Masked Diffusion Model \( M \) integrates the diffusion process with a Transformer encoder to model the reverse transitions:

\[
M(x_t, t) = \text{Linear}\left( \text{TransformerEncoder}\left( \text{PE}\left( E(x_{\text{masked}}) \right) \right) \right)
\]

where:
- \( E \) is the embedding function.
- \( x_{\text{masked}} \) incorporates masking based on the current time step \( t \).
- \( \text{PE} \) denotes positional encoding.
- \( \text{TransformerEncoder} \) represents the stack of Transformer encoder layers.

### Reward Model

The Reward Model \( R \) assigns scalar rewards to generated sequences based on their biological relevance:

\[
R(x) = \text{Linear}\left( \text{TransformerEncoder}\left( \text{PE}\left( E(x) \right) \right) \right) \in \mathbb{R}
\]

### Discriminator

The Discriminator \( D \) distinguishes between real and generated sequences, facilitating adversarial training:

\[
D(x) = \text{Linear}\left( \text{TransformerEncoder}\left( \text{PE}\left( E(x) \right) \right) \right) \in \mathbb{R}
\]

### Attention Visualizer

The Attention Visualizer computes and visualizes attention weights within Transformer layers:

\[
A = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right)
\]

These attention weights \( A \) elucidate the focus of the model on different regions of the input sequence, enabling interpretability of the generated sequences.

## Loss Functions

### Total Loss

The total optimization objective \( \mathcal{L} \) combines all loss components with respective weighting factors:

\[
\mathcal{L} = -\mathbb{E}[R(x)] + \alpha \mathcal{L}_{\text{KL}} + \beta \mathcal{L}_{\text{diversity}} + \gamma \mathcal{L}_{\text{adv}} + \delta \mathcal{L}_{\text{constraint}}
\]

where:
- \( \mathbb{E}[R(x)] \) encourages the generation of high-reward sequences.
- \( \mathcal{L}_{\text{KL}} \) aligns the generated distribution with the target distribution.
- \( \mathcal{L}_{\text{diversity}} \) promotes diversity among generated sequences.
- \( \mathcal{L}_{\text{adv}} \) ensures adversarial realism.
- \( \mathcal{L}_{\text{constraint}} \) enforces biological constraints.

## Training Optimization

### Gradient-Based Optimization

Parameters \( \theta \) are optimized using gradient descent:

\[
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
\]

where \( \eta \) is the learning rate.

### Learning Rate Scheduling

Adaptive learning rate schedulers modulate \( \eta \) over training iterations to enhance convergence:

- **OneCycleLR:**

\[
\eta(t) = \eta_{\text{max}} \cdot \sin^2\left( \frac{\pi t}{2T} \right)
\]

where \( t \) is the current iteration and \( T \) is the total number of iterations.

- **CosineAnnealingLR:**

\[
\eta(t) = \eta_{\text{min}} + \frac{1}{2} (\eta_{\text{max}} - \eta_{\text{min}}) \left(1 + \cos\left( \frac{\pi t}{T} \right)\right)
\]

- **ReduceLROnPlateau:**

Adjusts \( \eta \) based on the performance plateauing of a monitored metric.

### Alpha Adjustment

The weighting factor \( \alpha \) for the KL divergence is dynamically adjusted based on recent average rewards \( \bar{R} \):

\[
\alpha_{\text{new}} = \max\left( \frac{\alpha_{\text{initial}}}{\bar{R} + \epsilon}, \alpha_{\text{min}} \right)
\]

where \( \epsilon \) is a small constant to prevent division by zero, and \( \alpha_{\text{min}} \) ensures \( \alpha \) does not become excessively small.

## Data Augmentation

### Reverse Complementation

Data augmentation enhances training diversity by generating reverse complements of DNA sequences. For a DNA sequence \( S = s_1 s_2 \ldots s_L \), the reverse complement \( S' \) is defined as:

\[
S' = \overline{s_L} \, \overline{s_{L-1}} \, \ldots \, \overline{s_1}
\]

with nucleotide complementation rules:

\[
\overline{A} = T, \quad \overline{C} = G, \quad \overline{G} = C, \quad \overline{T} = A
\]

## Attention Mechanism

Within Transformer layers, attention weights \( A \) are computed as:

\[
A = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right)
\]

These weights determine the influence of each token in the sequence on others, enabling the model to capture intricate dependencies and focus on biologically significant motifs.

## Constraints and Memory Bank

### Constraint Enforcement

Constraints ensure that generated sequences adhere to specific biological requirements. For example, enforcing a target GC content involves minimizing the deviation between the actual and desired GC content:

\[
\mathcal{L}_{\text{gc}} = \left( \frac{\sum_{i=1}^{L} \mathbb{I}(x_i \in \{C, G\})}{L} - \text{GC}_{\text{target}} \right)^2
\]

### Memory Bank

A memory bank \( \mathcal{M} \) stores high-reward sequences to promote diversity and quality in generation:

\[
\mathcal{M} = \left\{ x \mid R(x) > \frac{1}{|\mathcal{W}|} \sum_{x' \in \mathcal{W}} R(x') \right\}
\]

where \( \mathcal{W} \) is a sliding window of recent rewards.

## Sequence Generation

### Sampling Trajectory

Generating a DNA sequence involves sampling a trajectory \( \{x_0, x_1, \ldots, x_T\} \) through the reverse diffusion process:

\[
x_{t-1} \sim p_\theta(x_{t-1} | x_t)
\]

Starting from \( x_T \sim p(x_T) \), the process iteratively denoises to produce \( x_0 \), the final generated sequence.

### Top-K and Top-P Sampling

To refine generation, Top-K and Top-P (nucleus) sampling techniques are employed:

- **Top-K Sampling:**

Restricts the sampling to the top \( k \) highest probability tokens:

\[
\pi'_{i} = \begin{cases}
\pi_i & \text{if } \pi_i \geq \text{TopK}_{k}(\pi) \\
-\infty & \text{otherwise}
\end{cases}
\]

- **Top-P Sampling:**

Retains the smallest set of tokens whose cumulative probability exceeds a threshold \( p \):

\[
\pi'_{i} = \begin{cases}
\pi_i & \text{if cumulative probability} \leq p \\
-\infty & \text{otherwise}
\end{cases}
\]

These techniques mitigate the risk of low-probability token generation, enhancing the quality and diversity of the output sequences.

## Optimization Objective

The optimization objective \( \mathcal{L} \) integrates multiple loss components to balance reward maximization, distribution alignment, diversity, adversarial realism, and constraint adherence:

\[
\mathcal{L} = -\mathbb{E}[R(x)] + \alpha \mathcal{L}_{\text{KL}} + \beta \mathcal{L}_{\text{diversity}} + \gamma \mathcal{L}_{\text{adv}} + \delta \mathcal{L}_{\text{constraint}}
\]

where:
- \( \mathbb{E}[R(x)] \) encourages the generation of high-reward sequences.
- \( \mathcal{L}_{\text{KL}} \) aligns the generated distribution with the target distribution.
- \( \mathcal{L}_{\text{diversity}} \) promotes diversity among generated sequences.
- \( \mathcal{L}_{\text{adv}} \) ensures adversarial realism.
- \( \mathcal{L}_{\text{constraint}} \) enforces biological constraints.
