# fasttext-langdetect
This library is a wrapper for the language detection model trained on fasttext by Facebook. For more information, please visit: https://fasttext.cc/docs/en/language-identification.html


## Supported languages
```
af als am an ar arz as ast av az azb ba bar bcl be bg bh bn bo bpy br bs bxr ca cbk ce cebckb co cs cv cy da de diq dsb dty dv el eml en eo es et eu fa fi fr frr fy ga gd gl gn gom gu gv he hi hif hr hsb ht hu hy ia id ie ilo io is it ja jbo jv ka kk km kn ko krc ku kv kw ky la lb lez li lmo lo lrc lt lv mai mg mhr min mk ml mn mr mrj ms mt mwl my myv mzn nah nap nds ne new nl nn no oc or os pa pam pfl pl pms pnb ps pt qu rm ro ru rue sa sah sc scn sco sd sh si sk sl so sq sr su sv sw ta te tg th tk tl tr tt tyv ug uk ur uz vec vep vi vls vo wa war wuu xal xmf yi yo yue zh
```

## Benchmark
We benchmarked the fasttext model against [cld2](https://github.com/CLD2Owners/cld2), [langid](https://github.com/saffsd/langid.py), and [langdetect](https://github.com/Mimino666/langdetect) on Wili-2018 dataset.

|                          | fasttext    | langid      | langdetect  | cld2        |
|--------------------------|-------------|-------------|-------------|-------------|
| Average time (ms) | 0,158273381 | 1,726618705 | 12,44604317 | **0,028776978** |
| 139 langs - not weighted   | 76,8        | 61,6        | 37,6        | **80,8**        |
| 139 langs - pop weighted | **95,5**        | 93,1        | 86,6        | 92,7        |
| 44 langs - not weighted    | **93,3**        | 89,2        | 81,6        | 91,5        |
| 44 langs - pop weighted   | **96,6**        | 94,8        | 89,4        | 93,4        |

- `pop weighted` means recall for each language is multipled by [its number of speakers](https://en.wikipedia.org/wiki/List_of_languages_by_total_number_of_speakers).
- 139 languages = all languages with ISO 639-1 2-letter code
- 44 languages = top 44 languages spoken in the world

## Install
```
pip install fasttext-langdetect
```

## Usage
`detect` method expects UTF-8 data. `low_memory` option enables getting predictions with the compressed version of the fasttext model by sacrificing the accuracy a bit.

```
from ftlangdetect import detect

result = detect(text="Bugün hava çok güzel", low_memory=False)
print(result)
> {'lang': 'tr', 'score': 1.00}

result = detect(text="Bugün hava çok güzel", low_memory=True)
print(result)
> {'lang': 'tr', 'score': 0.9982126951217651}
```

## References
[1] A. Joulin, E. Grave, P. Bojanowski, T. Mikolov, [Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)

```
@article{joulin2016bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}
```

[2] A. Joulin, E. Grave, P. Bojanowski, M. Douze, H. Jégou, T. Mikolov, [FastText.zip: Compressing text classification models](https://arxiv.org/abs/1612.03651)

```
@article{joulin2016fasttext,
  title={FastText.zip: Compressing text classification models},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1612.03651},
  year={2016}
}
```
