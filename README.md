# Vision Transformer ile Ateş Tespiti

Geleneksel nesne algılama yöntemlerinin ötesine geçen bir yaklaşım olan Vision Transformer (ViT) ile DINO modelini kullanarak gerçekleştirdiğim nesne algılama projesi, ateş tespitini sağlar. Kendi veri setimi kullanarak DINO modelini ateş tespiti gerçekleştirecek şekilde eğittim.

## Projeyi yükleyin

```bash
    git clone https://github.com/VuralBayrakli/vision-transformer-detection.git
```

## Modeli indirin
Eğitilmiş modeli [buradan](https://drive.google.com/file/d/1-V9aQpRVKR_pZO24K7DX2kALuqSpTGTo) indirin ve projenin kök dizinine yerleştirin.


## Gerekli kütüphaneleri yükleyin
```bash
    pip install -r requirements.txt
```

## Modeli kullanın
```bash
    python inference.py --model <model_path> --image <image_path>
```
