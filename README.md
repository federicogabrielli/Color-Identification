# 🎨 Advanced Image Color Palette Extraction

This project extracts the dominant colors from an image using a **multi-stage clustering pipeline** combined with a **perceptual density-based selection method**.
The goal is to obtain **visually meaningful color palettes**, not just mathematically averaged ones.

---

# 🧠 What This Project Does

Given an image, the algorithm:

1. Analyzes all its pixels
2. Groups similar colors together
3. Merges those groups into a few dominant colors
4. Selects the most representative color in each group
5. Displays the result as a clean color bar overlay

---

# 🔬 Step-by-Step Explanation

## 1. Image Loading and Preprocessing

```python
img = Image.open(percorso).convert("RGB")
img_small = img.resize((200, 200))
```

* The image is converted to RGB
* It is resized to **200×200 pixels**

### Why resize?

* Faster computation
* Removes unnecessary noise
* Keeps overall color distribution intact

---

## 2. Flattening the Image into Pixels

```python
pixels_rgb = np.array(img_small)
pixels_rgb_flat = pixels_rgb.reshape(-1, 3)
```

The image becomes a list of pixels:

```
[N_pixels × 3]  → (R, G, B)
```

---

## 3. Conversion to LAB Color Space

```python
pixels_lab = rgb2lab(pixels_rgb / 255.0)
```

We convert RGB → LAB.

### Why LAB?

* More aligned with human vision
* Distances between colors are more meaningful
* Improves clustering quality

---

## 4. First Clustering (Detailed Colors)

```python
kmeans = KMeans(n_clusters=k_iniziale)
labels = kmeans.fit_predict(pixels_lab)
```

* Groups pixels into many clusters (e.g. 10–12)
* Captures fine color variations

Then we compute:

```python
counts = np.bincount(labels)
percentuali = counts / len(labels)
```

→ percentage of each cluster in the image

---

## 5. Second Clustering (Color Merging)

```python
kmeans2 = KMeans(n_clusters=n_finale)
labels2 = kmeans2.fit_predict(kmeans.cluster_centers_)
```

Instead of clustering pixels again, we cluster **the cluster centers**.

### Why?

* Merges similar colors
* Produces a cleaner palette
* Reduces noise

---

## 6. Mapping Pixels to Final Clusters

```python
cluster_pixel_map = labels2[labels]
```

Each pixel is now assigned to a **final color group**.

---

## 7. Dominant Color Selection (Core Idea)

For each final cluster:

```python
sample = pixels_cluster[np.random.choice(...)]
dist = pairwise_distances(sample)
density = np.sum(dist < threshold, axis=1)
dominant_color = sample[np.argmax(density)]
```

### What this means

We:

1. Sample pixels from the cluster
2. Measure distances between them
3. Count how many neighbors each pixel has
4. Select the pixel with the highest local density

---


## 8. Percentage Calculation

```python
percentuali_finali.append(
    percentuali[np.where(labels2 == i)[0]].sum() * 100
)
```

Each final color gets a percentage based on:

* how many pixels belong to it

---

## 9. Sorting Colors

```python
idx = np.argsort(percentuali_finali)[::-1]
```

Colors are ordered from most dominant → least dominant.

---

# 🎬 Visualization

The output shows:

* The original image
* A horizontal color bar overlay
* Each segment:

  * represents a color
  * has width proportional to its percentage
  * displays its percentage inside

This creates a **cinematic palette visualization**, similar to color analysis tools used in film and design.

---

# ⚙️ Parameters

```python
k_iniziale = 15   # initial clusters
n_finale = 3      # final colors
```

### How they affect results

* Higher `k_iniziale` → more detail
* Lower `n_finale` → cleaner palette
* Higher `n_finale` → more nuanced palette

---

# 📸 Example Output

* Clean dominant colors
* Accurate percentages
* Visually meaningful palette

---

# 💡 Possible Improvements

* Merge very similar colors (perceptual threshold)
* Export HEX palettes
* Apply to video frames (real-time palette extraction)

---

# 👨‍💻 Author

This project explores:

* color perception
* clustering algorithms
* visual representation of data

---

# ⭐️ If you find this useful

Feel free to star the repository or build on top of it!
