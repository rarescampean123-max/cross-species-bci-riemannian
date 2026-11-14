# Cross-Species BCI: Decodarea Anticiparii Recompensei prin Geometrie Riemanniana si Aliniere Euclidiana

Acest repository contine implementarea experimentala si documentatia aferenta proiectului de cercetare privind **transferul de invatare trans-specie (Om -> Caine)** in Interfetele Creier-Computer (BCI), utilizand **Geometria Riemanniana** si **Alinierea Euclidiana (EA)** in Matricea Identitate ($I$).

---

## 1. Obiectivul Cercetarii

Proiectul investigheaza conservarea evolutiva a circuitului fronto-striatal (in special **nucleul caudat**) in timpul sarcinilor de **anticipare a recompensei**. Prin proiectarea matricilor de covarianta spatiala pe varietatea matricilor simetrice si pozitiv definite (SPD) si alinierea acestora la un punct de referinta comun ($I$), se elimina variabilitatea anatomica specifica fiecarei specii (domain shift).

### Ipoteze de Lucru
* **H1 (Conservarea Evolutiva):** Tiparele de conectivitate functionala din nucleul caudat la anticiparea recompensei prezinta o structura topologica similara la om si caine dupa alinierea in spatiul Riemannian.
* **H2 (Eficienta Alinierii Euclidiene):** Alinierea Euclidiana (EA) la Matricea Identitate ($I$) permite transferul unui clasificator antrenat exclusiv pe date umane catre date canine (*Zero-Shot Cross-Species Decoding*), pastrand intacta discriminabilitatea starilor mentale.

---

## 2. Structura Proiectului

```text
cross-species-bci-riemannian/
├── data/                          # Seturi de date preprocesate si matrici SPD (ignorate in Git)
├── docs/                          # Documentatia cercetarii si raportul final (.docx)
│   └── Idee_Proiect_Cercetare.docx
├── results/                       # Grafice, matrici de confuzie si tabele cu metrici
├── src/                           # Codul sursa al pipeline-ului experimental
│   ├── 01_preprocessing_roi.py    # Filtrare spectrala si extragere ROI (Nucleul Caudat)
│   └── 02_riemannian_alignment.py # Matricile de covarianta SPD si Alinierea Euclidiana in I
├── .gitignore
├── README.md
└── requirements.txt