# Vision Transformer ile Ateş Tespiti

Geleneksel nesne algılama yöntemlerinin ötesine geçen bir yaklaşım olan Vision Transformers yapısını içeren DINO modelini kullanarak gerçekleştirdiğim bu projede projede ateş tespitini sağlamaya çalıştım. Kendi veri setimi kullanarak DINO modelini ateş tespiti gerçekleştirecek şekilde eğittim.

## Projeyi yükleyin

```bash
    git clone https://github.com/VuralBayrakli/vision-transformer-detection.git
```

## DINO projesini indirin ve projenin kök dizine ekleyin
```bash
    git clone https://github.com/IDEA-Research/DINO.git
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

## Test Görüntüleri
![App Screenshot](https://github.com/VuralBayrakli/vision-transformer-detection/blob/master/results/res1.png)

![App Screenshot](https://github.com/VuralBayrakli/vision-transformer-detection/blob/master/results/res2.png)

![App Screenshot](https://github.com/VuralBayrakli/vision-transformer-detection/blob/master/results/res3.png)

![App Screenshot](https://github.com/VuralBayrakli/vision-transformer-detection/blob/master/results/res4.png)

![App Screenshot](https://github.com/VuralBayrakli/vision-transformer-detection/blob/master/results/res5.png)

![App Screenshot](https://github.com/VuralBayrakli/vision-transformer-detection/blob/master/results/res6.png)
