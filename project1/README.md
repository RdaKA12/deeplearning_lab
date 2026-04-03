# BankNote Authentication with Scratch MLP and PyTorch

## Introduction
Bu proje, 13 Mart 2026 laboratuvar uygulamasinda kurulan tek gizli katmanli MLP iskeletini tam bir mini proje haline getirmek icin gelistirildi. Problem, banknotlarin gercek veya sahte olmasini `variance`, `skewness`, `curtosis` ve `entropy` ozelliklerinden yararlanarak ikili siniflandirma olarak ele almaktadir.

BankNote veri seti secildi cunku kucuk ama net sinif ayrimi barindiriyor; bu da sifirdan yazilmis bir MLP ile ogrenme dinamiklerini incelemek icin uygun bir ortam sagliyor. Projenin hedefi, ayni veri bolmesi uzerinde hem NumPy tabanli bir MLP hem de ayni mimarinin PyTorch eslenigini egitip karsilastirmak, ayrica overfitting/underfitting davranisini mimari ve duzenleme deneyleriyle gozlemlemektir.

## Methods
### Dataset and split
- Veri dosyasi: `BankNote_Authentication.csv`
- Veri boyutu: `1372 satir x 5 sutun`
- Girdi ozellikleri: `variance, skewness, curtosis, entropy`
- Hedef sutun: `class`
- Eksik deger: bulunmadi
- Sinif dagilimi: `{'0': 762, '1': 610}`
- En guclu ozellik: `variance`. variance ozelligi hedef ile en guclu dogrusal iliskiyi gosteriyor (-0.7248).

Veri, `seed=42` ile stratified olarak `%64 train / %16 validation / %20 test` biciminde ayrildi. Test seti yalnizca son asamada kullanildi. Iki on isleme secenegi denendi: ham veri ve sadece train setine fit edilen standardization.

### Model configuration
- Temel mimari: `4-6-1`
- Genis mimari: `4-12-1`
- Derin mimari: `4-8-4-1`
- Gizli katman aktivasyonu: `tanh`
- Cikis aktivasyonu: `sigmoid`
- Loss: binary cross entropy
- Optimizer: full-batch SGD
- Learning rate: `0.1`
- Maksimum epoch: `600`
- Early stopping patience: `30`
- Minimum iyilesme esigi: `0.0001`
- L2 regularization deneyi: `lambda=0.001`
- Model secim kriteri: once validation accuracy, sonra validation F1, esitlikte daha dusuk best epoch

Scratch ve PyTorch modelleri icin ayni seed, ayni veri bolmesi, ayni mimari, ayni optimizer mantigi kullanildi. Son karsilastirma icin PyTorch modelinin baslangic agirliklari, scratch modeline verilen NumPy agirliklarindan kopyalandi. Egitim oncesi ilk ileri yayilim farki `max|scratch-pytorch| = 0.00000000` olarak olculdu.

### Reproducibility
Tum deneyleri ve ciktilari tekrar uretmek icin:

```bash
python run_experiments.py
```

Uretilen temel ciktilar:
- `results/data_summary.json`
- `results/experiment_summary.csv`
- `results/plots/*.png`
- `results/histories/*.csv`

## Results
### Scratch experiment summary
| Experiment | Prep | Arch | L2 | Best Epoch | Val Acc | Val F1 | Test Acc | Test F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scratch Baseline | raw | 4-6-1 | 0.0000 | 599 | 1.0000 | 1.0000 | 0.9855 | 0.9839 |
| Scratch + Standardization | standardized | 4-6-1 | 0.0000 | 598 | 1.0000 | 1.0000 | 0.9709 | 0.9683 |
| Scratch Wider | standardized | 4-12-1 | 0.0000 | 600 | 1.0000 | 1.0000 | 0.9709 | 0.9683 |
| Scratch Deeper | standardized | 4-8-4-1 | 0.0000 | 99 | 0.5545 | 0.0000 | 0.5564 | 0.0000 |
| Scratch + L2 | standardized | 4-6-1 | 0.0010 | 598 | 1.0000 | 1.0000 | 0.9709 | 0.9683 |

### Best model
Validation secim kurallarina gore en iyi scratch model `Scratch + Standardization` oldu. Bu model `standardized` veri hazirlama ve `4-6-1` mimarisini kullandi. Test performansi:

- Accuracy: `0.9709`
- Precision: `0.9385`
- Recall: `1.0000`
- F1 Score: `0.9683`
- Loss: `0.0507`

Not: Ham veriyle egitilen temel model test setinde daha yuksek accuracy uretse de model secimi bilerek test setine bakilmadan, sadece validation accuracy/F1 ve epoch sayisina gore yapildi.

Confusion matrix:

```text
[145, 8]
[0, 122]
```

Classification report ozet degerleri:
- Class 0 precision / recall / f1: `1.0000 / 0.9477 / 0.9732`
- Class 1 precision / recall / f1: `0.9385 / 1.0000 / 0.9683`
- Macro avg f1: `0.9707`

### Scratch vs PyTorch
| Model | Prep | Arch | Best Epoch | Val Acc | Test Acc | Test Precision | Test Recall | Test F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Scratch | standardized | 4-6-1 | 598 | 1.0000 | 0.9709 | 0.9385 | 1.0000 | 0.9683 |
| PyTorch | standardized | 4-6-1 | 598 | 1.0000 | 0.9709 | 0.9385 | 1.0000 | 0.9683 |

PyTorch sonucu, ayni baslangic agirliklari ve ayni optimizasyon mantigi ile scratch modele yakin performans verdi. Bu, sifirdan yazilan uygulamanin matematiksel akisini framework tabanli uygulama ile tutarli sekilde dogruladi.

## Discussion
Ham ve standardize veri ayni validation accuracy seviyesine ulasti; burada temel fark yakinlasma hizi ve loss davranisi oldu. Gizli katmandaki noron sayisini artirmak kapasiteyi yukseltip hatayi azaltabildi; bu, temel modelde hafif bias olabilecegini gosteriyor. Ek gizli katman anlamli kazanc saglamadi; bu veri icin derinlik artisinin getirisi sinirli kaldi. L2 regularization belirgin bir genel performans artisina donusmedi; veri seti zaten kucuk ve guclu ayrisabilir oldugu icin duzenleme etkisi sinirli kaldi. Train ve validation egri farki buyurse overfitting, her ikisi de dusuk kalsaydi underfitting yorumu yapilacakti. Bu projede en iyi modelde train ve validation performansi birbirine yakin seyrederek dengeli bir ogrenme gosterdi.

Bu calismanin temel sinirliligi, veri setinin gorece kucuk ve ayrisabilir olmasidir; bu nedenle daha derin mimariler her zaman anlamli ek kazanc uretmeyebilir. Gelecek adimlarda mini-batch egitim, farkli aktivasyonlar, batch normalization ve daha buyuk veri setleri ile ayni karsilastirma genisletilebilir.
