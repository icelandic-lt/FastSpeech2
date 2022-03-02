from optparse import Values
import SequiturTool
from sequitur import Translator


def load_g2p(model_path):
    sequitur_options = Values()
    sequitur_options.modelFile = model_path
    sequitur_options.resume_from_checkpoint = False
    sequitur_options.shouldRampUp = False
    sequitur_options.trainSample = False
    sequitur_options.shouldTranspose = False
    sequitur_options.shouldSelfTest = False
    sequitur_options.newModelFile = False
    model = SequiturTool.procureModel(sequitur_options, None)
    if not model:
        print('Can\'t load g2p model.')
        sys.exit(1)
    return model


g2p = load_g2p("/models/g2p_talromur/ipd_clean_slt2018.mdl")


def translate(text):
    text = text.replace(",", " ,")
    text = text.replace(".", " .")
    text = text.replace("?", " ?")
    text = text.replace(":", " .")
    text = text.replace("\"", "")

    translator = Translator(g2p)
    phone = []
    for w in text.split(" "):
        try:
            if w in [".", ",", "?"]:
                phone.append("sp")
            if w == "<sp>":
                phone.append("sp")
            else:
                phones = translator(w.lower())
                phone.extend(phones)
            phone.append(" ")
        except Translator.TranslationFailure:
            pass
    return phone

if __name__ == "__main__":
    t = translate("Góðan daginn klukkan er korter yfir níu þann tíunda maí og í dag er sól og tíu gráður")
    print(translate("Halló ég kann að tala íslensku"))

