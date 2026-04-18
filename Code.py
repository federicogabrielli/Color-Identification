from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage.color import rgb2lab
from sklearn.metrics import pairwise_distances



def analizza_colori_avanzato(percorso, k_iniziale=12, n_finale=3):

    # =========================
    # 1. CARICAMENTO
    # =========================
    img = Image.open(percorso).convert("RGB")
    img_small = img.resize((200, 200))

    pixels_rgb = np.array(img_small)
    h, w, _ = pixels_rgb.shape

    pixels_rgb_flat = pixels_rgb.reshape(-1, 3)

    # =========================
    # 2. LAB + KMEANS 1
    # =========================
    pixels_lab = rgb2lab(pixels_rgb / 255.0).reshape(-1, 3)

    kmeans = KMeans(n_clusters=k_iniziale, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels_lab)

    counts = np.bincount(labels)
    percentuali = counts / len(labels)

    # =========================
    # 3. KMEANS 2 (cluster merging)
    # =========================
    kmeans2 = KMeans(n_clusters=n_finale, random_state=42, n_init=10)
    labels2 = kmeans2.fit_predict(kmeans.cluster_centers_)

    # =========================
    # 4. MAP PIXEL → CLUSTER FINALE
    # =========================
    cluster_pixel_map = labels2[labels]

    colori_finali = []
    percentuali_finali = []

    for i in range(n_finale):

        mask = (cluster_pixel_map == i)
        pixels_cluster = pixels_rgb_flat[mask]

        if len(pixels_cluster) == 0:
            continue

        # =========================
        # DOMINANTE VERO (FAST MODE)
        # =========================
        if len(pixels_cluster) > 50:
        
            # prendi un sottoinsieme per velocità
            sample = pixels_cluster[np.random.choice(len(pixels_cluster), min(200, len(pixels_cluster)), replace=False)]

            # distanza tra tutti i punti campionati
            dist = pairwise_distances(sample)

            # densità = quanti vicini entro soglia
            density = np.sum(dist < 25, axis=1)

            dominant_color = sample[np.argmax(density)]

        else:
            dominant_color = pixels_cluster[np.random.randint(len(pixels_cluster))]

        colori_finali.append(dominant_color)

        percentuali_finali.append(percentuali[np.where(labels2 == i)[0]].sum() * 100)

    colori_finali = np.array(colori_finali)
    percentuali_finali = np.array(percentuali_finali)

    # =========================
    # SORT
    # =========================
    idx = np.argsort(percentuali_finali)[::-1]
    colori_finali = colori_finali[idx]
    percentuali_finali = percentuali_finali[idx]

    return img, colori_finali, percentuali_finali


def plot_risultato(img, colori, percentuali):

    fig, ax = plt.subplots(figsize=(8, 6))

    # =========================
    # IMAGE
    # =========================
    ax.imshow(img)
    ax.axis("off")

    # ottieni dimensioni immagine
    h, w = img.size[1], img.size[0]

    # =========================
    # BAR OVERLAY (dentro immagine)
    # =========================
    current_x = 0

    bar_y = h * 0.92      # posizione verticale barra (dentro immagine)
    bar_h = h * 0.08      # altezza barra

    for colore, perc in zip(colori, percentuali):
        width = w * (perc / 100)

        ax.add_patch(plt.Rectangle(
            (current_x, bar_y),
            width,
            bar_h,
            color=np.array(colore) / 255,
            linewidth=0
        ))

        # testo percentuale
        text_color = "white" if np.mean(colore) < 140 else "black"

        ax.text(
            current_x + width / 2,
            bar_y + bar_h / 2,
            f"{int(round(perc))}%",
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=text_color
        )

        current_x += width

    # =========================
    # TITLE UNDER IMAGE
    # =========================
    ax.text(
        w / 2,
        h + 20,
        "2001: A Space Odissey",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="black"
    )

    # estendi limiti per includere titolo
    ax.set_xlim(0, w)
    ax.set_ylim(h + 60, 0)

    plt.tight_layout()
    plt.show()

# =========================
# RUN
# =========================
percorso = r"C:\Users\feder\OneDrive\Pictures\Catture di schermata\Screenshot 2026-04-14 210043.png"
img, colori, percentuali = analizza_colori_avanzato(
    percorso,
    k_iniziale=15,
    n_finale=3
)

plot_risultato(img, colori, percentuali)