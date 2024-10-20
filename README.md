# DRAKES Sequence Generator

## Overview

DRAKES integrates advanced mathematical frameworks to generate DNA sequences using diffusion processes, Transformer architectures, adversarial training, and reinforcement learning. The model employs stochastic differential equations, probabilistic transitions, and optimization algorithms to ensure the generation of biologically plausible and diverse DNA sequences.

## Mathematical Model

### Diffusion Process

The diffusion process transforms an initial data distribution \( q(x_0) \) into a target distribution \( p(x_T) \) through discrete time steps \( t = 1, 2, \ldots, T \). The forward diffusion introduces noise incrementally:

$$
x_t = \sqrt{\alpha_t} \, x_{t-1} + \sqrt{1 - \alpha_t} \, \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$

Parameters:
- \( x_0 \sim q(x_0) \): Initial data sample.
- \( \alpha_t \): Noise schedule parameter at time step \( t \).
- \( \epsilon_t \): Gaussian noise added at each step.

The reverse diffusion aims to reconstruct the original data by modeling the conditional distribution:

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))
$$

Where \( \mu_\theta \) and \( \Sigma_\theta \) are parameterized by neural networks to approximate the reverse transitions.

### Transformer Architecture

Transformers use self-attention mechanisms to capture dependencies within sequences. The core self-attention operation is defined as:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V
$$

Where:
- \( Q = X W_Q \), \( K = X W_K \), \( V = X W_V \): Query, Key, and Value matrices obtained by linear transformations of the input \( X \).
- \( d_k \): Dimensionality of the key vectors.

#### Multi-Head Attention

Multiple attention heads enable the model to focus on different representation subspaces:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

$$
\text{head}_i = \text{Attention}(Q W_{Q_i}, K W_{K_i}, V W_{V_i})
$$

Where:
- \( h \): Number of heads.
- \( W_O \): Output projection matrix.

#### Positional Encoding

Positional encodings inject sequence order information:

$$
\text{PE}(pos, 2i) = \sin\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)
$$

$$
\text{PE}(pos, 2i+1) = \cos\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right)
$$

Where:
- \( pos \): Position in the sequence.
- \( i \): Dimension index.
- \( d_{\text{model}} \): Model dimensionality.

### Gumbel-Softmax Trick

The Gumbel-Softmax enables differentiable sampling from categorical distributions, facilitating gradient-based optimization:

$$
y = \text{softmax}\left( \frac{\log(\pi) + g}{\tau} \right), \quad g \sim \text{Gumbel}(0,1)
$$

Where:
- \( \pi \): Logits.
- \( \tau \): Temperature parameter controlling distribution smoothness.

### Kullback-Leibler Divergence

KL Divergence measures the discrepancy between two probability distributions \( P \) and \( Q \):

$$
\text{KL}(P \| Q) = \sum_{x} P(x) \log\left( \frac{P(x)}{Q(x)} \right)
$$

Aggregated over diffusion steps:

$$
\mathcal{L}_{\text{KL}} = \sum_{t=1}^{T} \text{KL}\left( p_\theta(x_{t-1} | x_t) \| q(x_{t-1} | x_t) \right)
$$

### Diversity Loss

Promotes diversity in generated sequences by maximizing pairwise Hamming distances:

$$
\mathcal{L}_{\text{diversity}} = -\frac{1}{N(N-1)} \sum_{i \neq j} \frac{d_H(x_i, x_j)}{L}
$$

Where:
- \( N \): Batch size.
- \( L \): Sequence length.
- \( d_H(x_i, x_j) \): Hamming distance between sequences \( x_i \) and \( x_j \).

### Adversarial Loss

Encourages realism in generated sequences through adversarial training with a discriminator \( D \):

$$
\mathcal{L}_{\text{adv}} = \mathbb{E}_{x \sim P_{\text{gen}}} \left[ \log(1 - D(x)) \right]
$$

Where \( P_{\text{gen}} \) is the distribution of generated sequences.

### Constraint Loss

Ensures generated sequences meet biological constraints such as GC content:

$$
\mathcal{L}_{\text{constraint}} = \mathcal{L}_{\text{gc}} + \mathcal{L}_{\text{motif}}
$$

$$
\mathcal{L}_{\text{gc}} = \left( \frac{\sum_{i=1}^{L} \mathbb{I}(x_i \in \{C, G\})}{L} - \text{GC}_{\text{target}} \right)^2
$$

Where \( \mathbb{I} \) is the indicator function and \( \text{GC}_{\text{target}} \) is the desired GC content.

## Model Components

### Masked Diffusion Model

The Masked Diffusion Model \( M \) integrates the diffusion process with a Transformer encoder:

$$
M(x_t, t) = \text{Linear}\left( \text{TransformerEncoder}\left( \text{PE}\left( E(x_{\text{masked}}) \right) \right) \right)
$$

Where:
- \( E \): Embedding function.
- \( x_{\text{masked}} \): Input sequence with masks based on time step \( t \).
- \( \text{PE} \): Positional encoding.
- \( \text{TransformerEncoder} \): Stack of Transformer encoder layers.
