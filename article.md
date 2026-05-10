# Manifolds and the Geometry of Dynamics for Time Series Analysis Time series data often represent observations from complex systems
governed by non-linear, deterministic processes. Traditional linear...

### Manifolds and the Geometry of Dynamics for Time Series Analysis
Time series data often represent observations from complex systems governed by non-linear, deterministic processes. Traditional linear models fail to capture such intricacies, but manifold reconstruction methods, particularly the pioneering work by George Sugihara and colleagues, offer powerful tools for studying these systems. This chapter explores time series manifolds, embedding theory, and their applications in uncovering the geometry of dynamic systems.


<figcaption>Photo by <a class="markup--anchor markup--figure-anchor" rel="photo-creator noopener" target="_blank">Diane Helentjaris</a> on <a class="markup--anchor markup--figure-anchor"


#### The Geometry of Time Series: Manifolds and Attractors
In non-linear dynamics, systems evolve on structures called manifolds --- smooth, high-dimensional surfaces that describe the system's state space. Over time, these dynamics may collapse into an attractor, a lower-dimensional structure capturing the system's essential behavior.

Examples include:

- A dripping faucet's dynamics following a chaotic yet deterministic pattern represented by an attractor.
- Climate systems evolving on a manifold defined by interacting variables such as temperature, pressure, and ocean currents.

The central challenge lies in reconstructing these manifolds from one-dimensional time series data.

#### Takens' Embedding Theorem and State Space Reconstruction
Takens' Embedding Theorem demonstrates that it is possible to reconstruct the state space of a system using time-delayed observations of a single variable.

Under suitable conditions, the reconstructed manifold is topologically equivalent to the true state space, preserving the system's dynamics.

#### Nonlinear Forecasting and Causality
Building on manifold reconstruction, Sugihara developed practical tools for analyzing time series, emphasizing:

- **Nonlinear Forecasting**: Predicting system behavior using reconstructed manifolds.
- **Causal Inference**: Distinguishing causation from correlation by analyzing interactions between manifolds.

#### Convergent Cross Mapping (CCM)
A key innovation by Sugihara is Convergent Cross Mapping (CCM), a method for inferring causality from time series data. CCM exploits the geometry of reconstructed manifolds to determine whether one variable's dynamics are influenced by another.

**Principle:** If variable X drives Y, the dynamics of Y should encode information about X. This information can be extracted by:

1.  [Reconstructing Y's manifold.]
2.  [Using it to estimate X's values.]
3.  [Evaluating estimation accuracy to infer causality.]

### Advantages of Time Series Manifolds
Manifolds preserve the system's true geometry and capturing Nonlinear Dynamics --- something linear models can't do. Manifold approaches explicitly test causation rather than relying on correlation. The same math works for a bunch of applications.

**Ecology**

- Marine ecosystems: CCM uncovers predator-prey relationships, such as sardine and anchovy populations in the California Current.
- Biodiversity dynamics: Reconstructed manifolds reveal species interaction effects on population changes.

**Climate Science**

- El Niño-Southern Oscillation (ENSO): Manifolds help uncover causal links between ocean temperature anomalies and global climate patterns.
- Tipping points: Early warning signals for climate shifts emerge from changes in attractor geometry.

**Financial Systems**

- Nonlinear forecasting models market dynamics and identifies causal drivers of financial crises.

**Medicine and Physiology**

- Cardiac dynamics: Reconstructed manifolds of ECG data reveal arrhythmias.
- Neural systems: CCM distinguishes causative from incidental neural interactions in brain activity.

### Constructing and Analyzing Manifolds: Practical Steps
**Selecting Time Delay (τ):** Use the autocorrelation function or mutual information to determine an optimal delay that balances redundancy and information content.

**Choosing Embedding Dimension (m):** Apply methods like false nearest neighbors (FNN) to identify the minimal mm that unfolds the manifold without overlapping trajectories.

**Reconstruction:** Form the embedded vectors and visualize the reconstructed attractor using techniques like 3D phase plots.

**Validation:** Validate the reconstructed manifold using forecasting skill (e.g., predicting future states) or CCM results.

### Case Study: Sardine-Anchovy Dynamics
The classic example is using this to measure sardine and anchovy populations in the Pacific Ocean. Sugihara used historical population data to build separate manifolds for sardines and anchovies. Then Sugihara applied CMM to test whether changes in sardine abundance predict anchovy dynamics (and vice versa). CCM revealed a causal relationship where environmental conditions mediate sardine-anchovy cycles. The reconstructed manifolds provide a geometric interpretation of this interaction.

### Enough theory. Let's build
Ok. so what does this look like in practice? We can use Python to illustrate the key concepts of t manifolds and their analysis. We can do this all using common libraries: NumPy, pandas, scikit-learn, and matplotlib.

#### Example 1: Time-Delay Embedding and Attractor Reconstruction
The Time-Delay Embedding reconstructs the attractor of a chaotic system (e.g., Lorenz) from a single time series.



### False Nearest Neighbors for Embedding Dimension Selection
The False Nearest Neighbors (FNN) determines the optimal embedding dimension by minimizing false neighbors in the reconstructed space.



### Convergent Cross Mapping (CCM)
CCM infers causality by evaluating how well the dynamics of one variable can predict another.



### Example 2: Finance
This example applies time series manifold reconstruction and **CCM** to analyze causality in financial markets. Specifically, we investigate whether movements in a stock index (e.g., S&P 500) drive or are driven by a related financial indicator (e.g., bond yields).

We simulate two causally linked financial time series for this example. Replace these simulated series with real-world data for practical applications.



### Time-Delay Embedding
We reconstruct the manifold of each time series using time-delay embedding. This is necessary to analyze the geometry of the underlying dynamics.



### Convergent Cross Mapping (CCM)
CCM is applied to analyze whether movements in the stock index drive bond yields or vice versa.


**Bond Yield → Stock Index**: A high correlation in this direction implies that bond yields drive stock index dynamics.


**Stock Index → Bond Yield**: If the CCM correlation for this direction is high, it suggests that changes in the stock index influence bond yields.


### Challenges and Future Directions
**High-Dimensional Systems**

Reconstructing manifolds in high-dimensional systems quickly becomes computationally intensive as the number of dimensions grows. To overcome this challenge, researchers are focusing on creating more efficient algorithms and exploring advanced dimensionality reduction techniques to streamline the process.

**Noise and Data Quality**

Time series data from real-world scenarios often come with noise and missing values, complicating analysis. Developing robust methods to reconstruct manifolds in noisy environments remains a critical area of ongoing research, ensuring accurate insights despite imperfect data.

**Integrating Multivariate Data**

While many current applications center on single-variable reconstructions, extending manifold techniques to multivariate time series unlocks deeper insights into complex system dynamics. This approach holds the potential to reveal interconnected patterns across multiple variables, enriching our understanding of underlying processes.

### So what?
Time series manifolds are awesome. they come with limitations but the whole approach makes more intuitive sense than how a neural network works.
