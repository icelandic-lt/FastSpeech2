Info
---

This code has been adapted to run on Icelandic data and produce sinthesized speech in Icelandic.
The changes include a data loader in `data/talromur.py`, improved test and synthesize scripts and an icelandic G2P module. 

Requirements
---

Requirements to train a TTS model:

   - Speech data, [Talrómur](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/104)
   - Alignments for the speech data (TextGrid files) [Talrómur-utils](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/201)

Requirements to synthesize speech:
   - Trained TTS model
   - Vocoder model
   - G2P model

You can find baseline models in the [Talrómur-utils](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/201) bundle.


Examples
---

These steps preprocess the data, train a model, test the model and synthesizes a few utterances.
For more info look at the help for each script `python train.py --help`.
It is also helpful to look at the `hparams.py` file.

```bash
python preprocess.py

python train.py

python test.py --step <step> --model-melgan <vocoder-model>

python synthesize.py --step <step> --sentences <utterances> --model-g2p <sequter-g2p-model> --model-melgan <vocoder-model>
```
