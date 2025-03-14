import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

old_embeddings = model.get_input_embeddings().weight.detach().float().cpu().numpy()  # Convert to float32
new_embeddings = new_input_embed.detach().float().cpu().numpy()

common_indices = [new_token_to_id[token] for token in common_vocab]
new_indices = [new_token_to_id[token] for token in new_tokens]

all_embeddings = np.vstack([old_embeddings, new_embeddings])

# Reduce dimensionality to 2D using PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(all_embeddings)

old_reduced = reduced_embeddings[:len(old_embeddings)]
common_reduced = reduced_embeddings[len(old_embeddings):len(old_embeddings) + len(common_vocab)]
new_reduced = reduced_embeddings[len(old_embeddings) + len(common_vocab):]

# Subsample embeddings for better visualization
def subsample(embeddings, fraction=0.1):
    num_samples = int(len(embeddings) * fraction)
    indices = np.random.choice(len(embeddings), num_samples, replace=False)
    return embeddings[indices]

old_reduced_sampled = subsample(old_reduced, fraction=0.1)
common_reduced_sampled = subsample(common_reduced, fraction=0.1)
new_reduced_sampled = subsample(new_reduced, fraction=0.1)

plt.figure(figsize=(10, 8))
plt.scatter(old_reduced_sampled[:, 0], old_reduced_sampled[:, 1], c='blue', label='Old Embeddings', alpha=0.6, s=10)
plt.scatter(common_reduced_sampled[:, 0], common_reduced_sampled[:, 1], c='green', label='Common Embeddings', alpha=0.6, s=10)
plt.scatter(new_reduced_sampled[:, 0], new_reduced_sampled[:, 1], c='red', label='New Embeddings', alpha=0.6, s=10)

plt.title('2D Visualization of Embedding Differences')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()
