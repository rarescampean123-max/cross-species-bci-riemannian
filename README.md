# Cross-Species BCI: Decodarea Anticiparii Recompensei prin Geometrie Riemanniana si Aliniere Euclidiana

Acest repository contine implementarea experimentala si documentatia aferenta proiectului de cercetare privind **transferul de invatare trans-specie (Om -> Caine)** in Interfetele Creier-Computer (BCI), utilizand **Geometria Riemanniana** si **Alinierea Euclidiana (EA)** in Matricea Identitate ($I$).

---

## 1. Obiectivul Cercetarii

Proiectul investigheaza conservarea evolutiva a circuitului fronto-striatal (**Nucleul Caudat** si **Nucleus Accumbens**) in timpul sarcinilor de **anticipare a recompensei**. Prin proiectarea matricilor de covarianta spatiala pe varietatea matricilor simetrice si pozitiv definite (SPD) si alinierea acestora la un punct de referinta comun ($I$), se elimina variabilitatea anatomica specifica fiecarei specii (*domain shift*).

### Ipoteze de Lucru
- **H1 (Conservarea Evolutiva):** Tiparele de conectivitate functionala din circuitul striatal la anticiparea recompensei prezinta o structura topologica similara la om si caine dupa alinierea in spatiul Riemannian.
- **H2 (Eficienta Alinierii Euclidiene):** Alinierea Euclidiana (EA) la Matricea Identitate ($I$) permite transferul unui clasificator antrenat exclusiv pe date umane catre date canine (*Zero-Shot Cross-Species Decoding*), pastrand intacta discriminabilitatea starilor mentale.

---

## 2. Structura Proiectului

```text
cross-species-bci-riemannian/
├── data/                                    # Date brute fMRI, masti anatomice si matrici SPD
├── docs/                                    # Documentatia cercetarii si raportul final (.docx)
│   └── Raport_Cercetare_Cross_Species_BCI.docx
├── results/                                 # Metrici JSON si figurile generate pentru raport
│   ├── h1_validation_metrics.json           # Rezultatele numerice si testul de permutare
│   ├── fig1_riemannian_alignment_pca.png    # Proiectia PCA inainte si dupa Alinierea Euclidiana
│   ├── fig2_connectivity_fingerprints.png   # Amprentele topologice de conectivitate (Om vs. Caine)
│   └── fig3_geodesic_metrics_permutation.png # Distantele geodezice si distributia nula (N=1000)
├── src/                                     # Codul sursa al pipeline-ului experimental
│   ├── 01_preprocessing_roi.py              # Filtrare spectrala si extragere ROI (Studiu de caz)
│   ├── 02_riemannian_alignment.py           # Estimare matrici SPD si Aliniere Euclidiana in I
│   ├── 03_cross_species_eval.py             # Validare pe esantion real fMRI, test permutare si MDM
│   └── 04_generate_figures.py               # Generarea automata a vizualizarilor din raport
├── .gitignore
├── README.md
└── requirements.txt
