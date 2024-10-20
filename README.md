# DRAKES Sequence Generator

## Mathematical Model

### Diffusion Process

The diffusion process in DRAKES transforms an initial data distribution \( q(x_0) \) into a complex target distribution \( p(x_T) \) through a series of stochastic transitions over discrete time steps \( t = 1, 2, \ldots, T \). The forward diffusion process is defined as:

\[
x_t = \sqrt{\alpha_t} \, x_{t-1} + \sqrt{1 - \alpha_t} \, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
\]

where:
- \( x_0 \sim q(x_0) \) is the initial data sample.
- \( \alpha_t \) is the noise schedule parameter at time step \( t \).
- \( \epsilon_t \) represents Gaussian noise added at each step.

The reverse diffusion process aims to recover the original data distribution by modeling the conditional distribution:

\[
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
\]

where \( \mu_\theta \) and \( \Sigma_\theta \) are parameterized by neural networks, representing the mean and covariance of the reverse transition.

### Transformer Architecture

Transformers utilize self-attention mechanisms to model dependencies within sequences. The core self-attention operation is mathematically expressed as:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V
\]

where:
- \( Q = X W_Q \), \( K = X W_K \), \( V = X W_V \) are the query, key, and value matrices obtained by linear transformations of the input \( X \).
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

The Gumbel-Softmax allows differentiable sampling from categorical distributions, facilitating gradient-based optimization. Given logits \( \pi \), the Gumbel-Softmax sample \( y \) is computed as:

\[
y = \text{softmax}\left( \frac{\log(\pi) + g}{\tau} \right), \quad g \sim \text{Gumbel}(0, 1)
\]

where \( \tau \) is the temperature parameter controlling the smoothness of the distribution.

### Kullback-Leibler Divergence

The KL divergence measures the discrepancy between two probability distributions \( P \) and \( Q \):

\[
\text{KL}(P \| Q) = \sum_{x} P(x) \log\left( \frac{P(x)}{Q(x)} \right)
\]

In the context of DRAKES, the KL divergence between the learned transition rates \( p_\theta(x_{t-1} | x_t) \) and the true distribution \( q(x_{t-1} | x_t) \) is aggregated over all diffusion steps:

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

To ensure generated sequences adhere to biological constraints, the constraint loss \( \mathcal{L}_{\text{constraint}} \) incorporates terms such as GC content \( \mathcal{L}_{\text{gc}} \):

\[
\mathcal{L}_{\text{gc}} = \text{MSE}\left( \frac{\sum_{i=1}^{L} \mathbb{I}(x_i \in \{C, G\})}{L}, \text{GC}_{\text{target}} \right)
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

## Loss Functions

### Total Loss

The total loss \( \mathcal{L} \) combines all loss components with respective weighting factors:

\[
\mathcal{L} = -\mathbb{E}[R(x)] + \alpha \mathcal{L}_{\text{KL}} + \beta \mathcal{L}_{\text{diversity}} + \gamma \mathcal{L}_{\text{adv}} + \delta \mathcal{L}_{\text{constraint}}
\]

where:
- \( \alpha \), \( \beta \), \( \gamma \), and \( \delta \) are hyperparameters controlling the influence of each loss component.

## Training Optimization

### Gradient-Based Optimization

The parameters \( \theta \) are optimized using gradient descent:

\[
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}
\]

where \( \eta \) is the learning rate.

### Learning Rate Scheduling

Adaptive learning rate schedulers adjust \( \eta \) over training iterations. Common schedulers include:

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

Reduces \( \eta \) when a monitored metric has stopped improving.

### Alpha Adjustment

The weighting factor \( \alpha \) for the KL divergence is dynamically adjusted based on recent rewards \( \bar{R} \):

\[
\alpha_{\text{new}} = \max\left( \frac{\alpha_{\text{initial}}}{\bar{R} + \epsilon}, \alpha_{\text{min}} \right)
\]

where \( \epsilon \) is a small constant to prevent division by zero.

## Data Augmentation

### Reverse Complementation

To enhance training diversity, sequences are augmented by their reverse complements. For a DNA sequence \( S = s_1 s_2 \ldots s_L \), the reverse complement \( S' \) is:

\[
S' = \overline{s_L} \, \overline{s_{L-1}} \, \ldots \, \overline{s_1}
\]

with nucleotide complementation defined as:

\[
\overline{A} = T, \quad \overline{C} = G, \quad \overline{G} = C, \quad \overline{T} = A
\]

## Attention Mechanism

The attention weights \( A \) within Transformer layers are computed as:

\[
A = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right)
\]

where \( Q \), \( K \), and \( V \) are query, key, and value matrices derived from the input embeddings. These weights determine the influence of each token in the sequence on others, enabling the model to capture intricate dependencies.

## Constraints and Memory Bank

### Constraint Enforcement

Constraints ensure generated sequences meet biological criteria. For instance, GC content constraint \( \mathcal{L}_{\text{gc}} \) is formulated as:

\[
\mathcal{L}_{\text{gc}} = \left( \frac{\sum_{i=1}^{L} \mathbb{I}(x_i \in \{C, G\})}{L} - \text{GC}_{\text{target}} \right)^2
\]

### Memory Bank

A memory bank \( \mathcal{M} \) stores high-reward sequences to facilitate diversity and quality:

\[
\mathcal{M} = \left\{ x \mid R(x) > \frac{1}{|\mathcal{W}|} \sum_{x' \in \mathcal{W}} R(x') \right\}
\]

where \( \mathcal{W} \) is the recent window of rewards.

## Sequence Generation

### Sampling Trajectory

The trajectory \( \{x_0, x_1, \ldots, x_T\} \) of generated sequences is sampled through the reverse diffusion process:

\[
x_{t-1} \sim p_\theta(x_{t-1} | x_t)
\]

### Top-K and Top-P Sampling

To refine generation, Top-K and Top-P sampling techniques are employed:

- **Top-K Sampling:**

\[
\pi'_{i} = \begin{cases}
\pi_i & \text{if } \pi_i \geq \text{TopK}_{k}(\pi) \\
-\infty & \text{otherwise}
\end{cases}
\]

- **Top-P Sampling:**

\[
\pi'_{i} = \begin{cases}
\pi_i & \text{if cumulative probability} \leq p \\
-\infty & \text{otherwise}
\end{cases}
\]

where \( \text{TopK}_{k}(\pi) \) retains the top \( k \) logits, and cumulative probability \( p \) defines the nucleus threshold.

## Optimization Objective

The optimization objective \( \mathcal{L} \) integrates multiple loss components to balance reward maximization, distribution alignment, diversity, adversarial realism, and constraint adherence:

\[
\mathcal{L} = -\mathbb{E}[R(x)] + \alpha \mathcal{L}_{\text{KL}} + \beta \mathcal{L}_{\text{diversity}} + \gamma \mathcal{L}_{\text{adv}} + \delta \mathcal{L}_{\text{constraint}}
\]

where:
- \( \mathbb{E}[R(x)] \) encourages high-reward sequence generation.
- \( \mathcal{L}_{\text{KL}} \) aligns the generated distribution with the target distribution.
- \( \mathcal{L}_{\text{diversity}} \) promotes sequence diversity.
- \( \mathcal{L}_{\text{adv}} \) ensures adversarial realism.
- \( \mathcal{L}_{\text{constraint}} \) enforces biological constraints.