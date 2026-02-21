# Variational Autoencoders (VAE): A Complete Course-Style Guide

## 1. Introduction to Generative Models

Generative models aim to learn the underlying data distribution (p(x)) so that we can:

* Generate new realistic samples
* Learn meaningful latent representations
* Perform unsupervised learning
* Handle missing data and uncertainty

Common generative models include:

* Autoencoders (AE)
* Variational Autoencoders (VAE)
* GANs
* Diffusion Models
* Flow-based Models

A **Variational Autoencoder (VAE)** is a probabilistic generative model that combines:

* Neural networks
* Variational inference
* Latent variable modeling

It was introduced by **Diederik P. Kingma** and **Max Welling** (2013).

---

# 2. From Autoencoder to Variational Autoencoder

## 2.1 Classical Autoencoder (AE)

A standard autoencoder has two parts:

### Encoder

Maps input to latent space:

$$
z = f_\phi(x)
$$

### Decoder

Reconstructs input:

$$
\hat{x} = g_\theta(z)
$$

Loss function:

$$
\mathcal{L}_{AE} = |x - \hat{x}|^2
$$

### Limitation of Autoencoders

* Latent space is not structured
* Cannot sample meaningfully
* No probabilistic interpretation
* Overfitting to reconstruction

This is where VAEs improve the framework.

---

# 3. Core Idea of Variational Autoencoder (VAE)

Instead of encoding a data point into a **single vector**, VAE encodes it into a **probability distribution**.

### Key Idea:

$$
x \rightarrow q_\phi(z|x) \rightarrow p_\theta(x|z)
$$

Where:

* (q_\phi(z|x)): Encoder (Approximate posterior)
* (p_\theta(x|z)): Decoder (Likelihood model)
* (p(z)): Prior (usually Gaussian)

---

# 4. Probabilistic Formulation

## 4.1 Latent Variable Model

We assume:

$$
p_\theta(x) = \int p_\theta(x|z)p(z),dz
$$

Where:

* (z) is latent variable
* (p(z) = \mathcal{N}(0, I)) (standard normal prior)

The goal: maximize the data likelihood

$$
\log p_\theta(x)
$$

But this integral is **intractable**.

---

# 5. Variational Inference and ELBO

To solve intractability, we introduce an approximate posterior:

$$
q_\phi(z|x) \approx p_\theta(z|x)
$$

Using variational inference, we derive the Evidence Lower Bound (ELBO):

$$
\log p_\theta(x) \geq \mathcal{L}_{ELBO}(x)
$$

Where:

$$
\mathcal{L}*{ELBO} =
\mathbb{E}*{q_\phi(z|x)}[\log p_\theta(x|z)]

* D_{KL}(q_\phi(z|x) | p(z))
  $$

This is the **core loss function of VAE**.

---

# 6. VAE Loss Function (Complete Breakdown)

## 6.1 Final VAE Objective

$$
\mathcal{L}*{VAE} =
\underbrace{\mathbb{E}*{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{Reconstruction Loss}}
--------------------------------------------------------------------------------------

\underbrace{D_{KL}(q_\phi(z|x) | p(z))}_{\text{Regularization}}
$$

### 6.2 Reconstruction Term

Measures how well the decoder reconstructs input.

Common choices:

* MSE (for continuous data)
* Binary Cross Entropy (for images in [0,1])

$$
\mathcal{L}_{rec} = |x - \hat{x}|^2 \quad \text{or} \quad BCE(x, \hat{x})
$$

---

### 6.3 KL Divergence Term

$$
D_{KL}(q_\phi(z|x) | p(z))
$$

This term:

* Forces latent distribution close to prior (\mathcal{N}(0, I))
* Ensures smooth latent space
* Enables sampling

For Gaussian posterior:

$$
q_\phi(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))
$$

KL divergence closed form:

$$
D_{KL} = \frac{1}{2} \sum ( \mu^2 + \sigma^2 - \log \sigma^2 - 1 )
$$

---

# 7. Gaussian VAE (Standard VAE Architecture)

## 7.1 Assumptions

Most VAEs assume:

* Prior: (p(z) = \mathcal{N}(0, I))
* Posterior: (q(z|x) = \mathcal{N}(\mu(x), \Sigma(x)))

Usually diagonal covariance:

$$
\Sigma = \text{diag}(\sigma^2)
$$

---

## 7.2 Encoder Network (Gaussian Encoder)

Instead of outputting a single vector, the encoder outputs:

* Mean: (\mu(x))
* Log-variance: (\log \sigma^2(x))

Architecture:

```
Input x → Neural Network → (μ(x), logσ²(x))
```

Why log variance?

* Numerical stability
* Ensures positive variance

---

# 8. The Reparameterization Trick (Critical Concept)

## 8.1 The Problem

Sampling:

$$
z \sim \mathcal{N}(\mu, \sigma^2)
$$

breaks backpropagation (non-differentiable sampling).

## 8.2 The Solution

Reparameterize:

$$
z = \mu + \sigma \cdot \epsilon
$$

$$
\epsilon \sim \mathcal{N}(0, I)
$$

Now gradients can flow through:

* (\mu)
* (\sigma)

This is the key innovation that makes VAEs trainable.

---

# 9. Full VAE Network Architecture

## 9.1 End-to-End Pipeline

### Step 1: Encoder

$$
x \rightarrow (\mu, \log \sigma^2)
$$

### Step 2: Sampling (Reparameterization)

$$
z = \mu + \sigma \cdot \epsilon
$$

### Step 3: Decoder

$$
z \rightarrow \hat{x}
$$

---

## 9.2 Typical Gaussian VAE Structure (for images)

### Encoder

* Input layer
* Conv / Linear layers
* Two output heads:

  * Mean head
  * Log-variance head

### Latent Space

* Dimension: 2–512 (task dependent)

### Decoder

* Fully connected / Transposed Conv
* Outputs reconstructed image

---

# 10. Mathematical Summary of Training

### Forward Pass

1. Encode:

$$
(\mu, \log\sigma^2) = Encoder(x)
$$

2. Sample:

$$
z = \mu + \sigma \cdot \epsilon
$$

3. Decode:

$$
\hat{x} = Decoder(z)
$$

### Loss Computation

$$
\mathcal{L} =
\text{Reconstruction Loss}
+
\beta \cdot D_{KL}
$$

Where:

* (\beta = 1) for standard VAE
* (\beta > 1) for β-VAE (stronger disentanglement)

---

# 11. Why Gaussian Prior?

Using (\mathcal{N}(0, I)) provides:

* Smooth latent space
* Easy sampling
* Analytical KL divergence
* Stable training

It also ensures:

$$
z \sim \text{continuous, structured manifold}
$$

---

# 12. Sampling and Generation

After training:

1. Sample latent vector:

$$
z \sim \mathcal{N}(0, I)
$$

2. Generate:

$$
x_{new} = Decoder(z)
$$

This allows:

* New image synthesis
* Interpolation in latent space
* Style transfer

---

# 13. Latent Space Properties

A well-trained VAE latent space is:

* Continuous
* Smooth
* Disentangled (in some variants)
* Interpretable (with proper design)

Interpolation:

$$
z_{interp} = \alpha z_1 + (1-\alpha) z_2
$$

Produces smooth transitions in generated outputs.

---

# 14. Variants of VAE (Advanced)

## 14.1 β-VAE

$$
\mathcal{L} = \mathcal{L}*{rec} + \beta D*{KL}
$$

* Better disentanglement
* Used in representation learning

## 14.2 Conditional VAE (CVAE)

Condition on label (y):

$$
q(z|x,y), \quad p(x|z,y)
$$

## 14.3 VQ-VAE

* Discrete latent variables
* Used in modern generative models

## 14.4 Hierarchical VAE

* Multiple latent layers
* Better expressiveness

---

# 15. Comparison: AE vs VAE

| Feature            | Autoencoder   | VAE           |
| ------------------ | ------------- | ------------- |
| Latent Space       | Deterministic | Probabilistic |
| Sampling           | Poor          | Excellent     |
| Generative Ability | Weak          | Strong        |
| Regularization     | None          | KL Divergence |
| Theory             | Heuristic     | Bayesian      |

---

# 16. Common Training Challenges

### 16.1 KL Collapse

* Decoder ignores latent variable
* Fix: KL annealing / β tuning

### 16.2 Blurry Reconstructions

Cause:

* Gaussian likelihood assumption
* Pixel-wise loss

### 16.3 Posterior Collapse

Fixes:

* Free bits
* KL warm-up
* Stronger encoder

---

# 17. Practical Implementation Tips

* Use log-variance instead of variance
* Normalize inputs (0–1)
* Latent dim: start with 16–64
* Use KL annealing for stability
* Monitor:

  * Reconstruction loss
  * KL loss separately

---

# 18. Final Intuition (One-Sentence)

A Variational Autoencoder is a probabilistic autoencoder that learns a **Gaussian latent distribution** instead of a fixed vector, optimized using the **ELBO loss (Reconstruction + KL divergence)** and trained via the **reparameterization trick** to enable end-to-end gradient-based learning.

---

# 19. Minimal Mathematical Summary Cheat Sheet

$$
q_\phi(z|x) = \mathcal{N}(\mu(x), \sigma^2(x))
$$

$$
z = \mu + \sigma \cdot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

$$
\mathcal{L}*{VAE} =
\mathbb{E}*{q(z|x)}[\log p(x|z)]

* D_{KL}(q(z|x) | \mathcal{N}(0, I))
  $$

This is the complete foundation of a Gaussian VAE.
