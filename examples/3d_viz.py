import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd

old_embeddings = model.get_input_embeddings().weight.detach().float().cpu().numpy()  # Convert to float32
new_embeddings = new_input_embed.detach().float().cpu().numpy()

common_indices = [new_token_to_id[token] for token in common_vocab]
new_indices = [new_token_to_id[token] for token in new_tokens]

all_embeddings = np.vstack([old_embeddings, new_embeddings])

pca = PCA(n_components=3)
reduced_embeddings = pca.fit_transform(all_embeddings)

old_reduced = reduced_embeddings[:len(old_embeddings)]
common_reduced = reduced_embeddings[len(old_embeddings):len(old_embeddings) + len(common_vocab)]
new_reduced = reduced_embeddings[len(old_embeddings) + len(common_vocab):]

def subsample(embeddings, fraction=0.1):
    num_samples = int(len(embeddings) * fraction)
    indices = np.random.choice(len(embeddings), num_samples, replace=False)
    return embeddings[indices]

old_reduced_sampled = subsample(old_reduced, fraction=0.1)
common_reduced_sampled = subsample(common_reduced, fraction=0.1)
new_reduced_sampled = subsample(new_reduced, fraction=0.1)

labels = (['Old Embeddings'] * len(old_reduced_sampled) +
          ['Common Embeddings'] * len(common_reduced_sampled) +
          ['New Embeddings'] * len(new_reduced_sampled))

all_reduced = np.vstack([old_reduced_sampled, common_reduced_sampled, new_reduced_sampled])

df = pd.DataFrame(all_reduced, columns=['PCA Component 1', 'PCA Component 2', 'PCA Component 3'])
df['Label'] = labels

# Define a custom color map
color_map = {
    'Old Embeddings': 'blue - old embedding',
    'Common Embeddings': 'green - common embedding',
    'New Embeddings': 'red - new embedding'
}
df['Color'] = df['Label'].map(color_map)

# Plot using Plotly
fig = px.scatter_3d(
    df,
    x='PCA Component 1',
    y='PCA Component 2',
    z='PCA Component 3',
    color='Color',
    title='3D Visualization of Embedding Differences',
    labels={'Color': 'Embedding Type'},
    opacity=0.5,
    width=1000,
    height=800
)

fig.update_traces(marker=dict(size=5))

fig.show()
