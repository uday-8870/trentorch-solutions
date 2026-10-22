# center-features-for-pca — Apple case

Beginner | problemset | unsupervised-ml | pca | apple

Apple-inspired on-device analytics pipeline is preparing correlated sensor features for dimensionality reduction. The team needs each feature centered before PCA, and you need to compute that centered matrix without changing the original observations beyond subtracting their means.

Subtract each column's mean from that column. Nothing else about the data changes.

Implement `solve(X)`.

**Returns.** Return a float NumPy matrix of the same shape in which every column has mean $0$.

### Examples

**Example 1**

Input:

```python
solve([[1, 2], [3, 4]])
```

Output:

```text
[[-1.0, -1.0], [1.0, 1.0]]
```

**Example 2**

Input:

```python
solve([[2.0, 5.0], [4.0, 5.0], [6.0, 5.0]])
```

Output:

```text
[[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]]
```
